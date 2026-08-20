//! omfx fork-owned: OpenAI-compatible wire codecs.
//!
//! This module converts the internal conversation types into the two
//! OpenAI-compatible request shapes fx talks directly to (the OpenAI Responses
//! API and the xAI Chat Completions API), and decodes their SSE streams back
//! into `types.GatewayCompletion`. It mirrors the message semantics of the
//! Vercel gateway path in `src/core/gateway/gateway_json.zig` so both routes
//! produce the same internal contract.
//!
//! Ownership: request builders return bytes owned by the caller. Stream
//! consumers return a completion whose `content`, `generation_id`,
//! `provider_failure_detail`, and `tool_calls` are allocated with the passed
//! allocator and never borrow the reader buffer.

const std = @import("std");

const agent_stream_provider = @import("../core/agent/stream_provider.zig");
const image_attachments = @import("../core/images/image_attachments.zig");
const io_mod = @import("../core/shared/io.zig");
const model_capabilities = @import("../core/config/model_capabilities.zig");
const types = @import("../core/shared/types.zig");

const Allocator = std.mem.Allocator;

/// Upper bound for a single SSE line before the stream is rejected.
const max_sse_event_line_bytes: usize = 32 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Request builders
// ---------------------------------------------------------------------------

/// Structured-output contract, mirroring
/// `agent_stream_provider.StructuredResponseFormat`.
pub const StructuredResponseFormat = struct {
    name: []const u8,
    description: []const u8 = "",
    schema_json: []const u8,
};

/// Every request knob both direct wires understand. The provider router owns
/// which combination it sends; the builders only enforce wire-level invariants.
pub const BuildOptions = struct {
    provider_options: model_capabilities.ResolvedProviderOptions = .{},
    tool_choice: types.ToolChoice = .auto,
    max_output_tokens: ?u32 = null,
    /// Requests JSON-schema constrained output.
    response_format: ?StructuredResponseFormat = null,
    /// Pre-verified image payloads for the final user message. When set, that
    /// message must be a user message that carries no inline attachments, the
    /// same invariant the gateway verified-images builder enforces.
    verified_images: ?[]const image_attachments.VerifiedSnapshot = null,
    /// Forces exactly this function call, overriding `tool_choice`.
    required_tool_name: ?[]const u8 = null,
};

/// Builds a streaming OpenAI Responses API request body. `model` is the
/// provider-native id, `serialized_tools` is the gateway tool array
/// (`[{"name":..,"description":..,"inputSchema":{..}}]`). The caller owns the
/// returned bytes.
pub fn buildResponsesRequestBody(
    alloc: Allocator,
    model: []const u8,
    serialized_tools: []const u8,
    messages: []const types.ChatMessage,
    options: BuildOptions,
) ![]u8 {
    const verified_index = try verifiedImageIndex(messages, options.verified_images);

    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    const writer = &out.writer;

    try writer.writeAll("{\"model\":");
    try std.json.Stringify.value(model, .{}, writer);

    const instructions = try joinSystemInstructions(alloc, messages);
    defer alloc.free(instructions);
    if (instructions.len > 0) {
        try writer.writeAll(",\"instructions\":");
        try std.json.Stringify.value(instructions, .{}, writer);
    }

    try writer.writeAll(",\"input\":[");
    var wrote_input = false;
    for (messages, 0..) |message, i| {
        switch (message.role) {
            .system => {},
            .user => {
                if (wrote_input) try writer.writeByte(',');
                try writeResponsesUserMessage(
                    alloc,
                    writer,
                    message,
                    if (verified_index == i) options.verified_images else null,
                );
                wrote_input = true;
            },
            .assistant => {
                if (message.content) |content| {
                    if (content.len > 0) {
                        if (wrote_input) try writer.writeByte(',');
                        try writer.writeAll("{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":");
                        try std.json.Stringify.value(content, .{}, writer);
                        try writer.writeAll("}]}");
                        wrote_input = true;
                    }
                }
                for (message.tool_calls) |call| {
                    if (wrote_input) try writer.writeByte(',');
                    try writer.writeAll("{\"type\":\"function_call\",\"call_id\":");
                    try std.json.Stringify.value(call.id, .{}, writer);
                    try writer.writeAll(",\"name\":");
                    try std.json.Stringify.value(call.name, .{}, writer);
                    try writer.writeAll(",\"arguments\":");
                    try std.json.Stringify.value(call.arguments_json, .{}, writer);
                    try writer.writeByte('}');
                    wrote_input = true;
                }
            },
            .tool => {
                if (wrote_input) try writer.writeByte(',');
                try writer.writeAll("{\"type\":\"function_call_output\",\"call_id\":");
                try std.json.Stringify.value(message.tool_call_id orelse "", .{}, writer);
                try writer.writeAll(",\"output\":");
                try std.json.Stringify.value(message.content orelse "", .{}, writer);
                try writer.writeByte('}');
                wrote_input = true;
            },
        }
    }
    try writer.writeByte(']');

    try writer.writeAll(",\"tools\":");
    try writeResponsesTools(alloc, writer, serialized_tools);

    try writer.writeAll(",\"tool_choice\":");
    if (options.required_tool_name) |name| {
        if (name.len == 0) return error.InvalidRequiredToolName;
        try writer.writeAll("{\"type\":\"function\",\"name\":");
        try std.json.Stringify.value(name, .{}, writer);
        try writer.writeByte('}');
    } else {
        try std.json.Stringify.value(options.tool_choice.label(), .{}, writer);
    }

    if (options.provider_options.parallel_tool_calls) |parallel| {
        try writer.writeAll(",\"parallel_tool_calls\":");
        try writer.writeAll(if (parallel) "true" else "false");
    }

    if (options.provider_options.reasoning) |*reasoning| {
        if (namedReasoningEffort(reasoning)) |effort| {
            try writer.writeAll(",\"reasoning\":{\"effort\":");
            try std.json.Stringify.value(effort, .{}, writer);
            try writer.writeByte('}');
        }
    }

    if (options.response_format) |format| {
        var schema = try parseStructuredSchema(alloc, format.schema_json);
        defer schema.deinit();
        try writer.writeAll(",\"text\":{\"format\":{\"type\":\"json_schema\",\"name\":");
        try std.json.Stringify.value(format.name, .{}, writer);
        if (format.description.len > 0) {
            try writer.writeAll(",\"description\":");
            try std.json.Stringify.value(format.description, .{}, writer);
        }
        try writer.writeAll(",\"strict\":false,\"schema\":");
        try std.json.Stringify.value(schema.value, .{}, writer);
        try writer.writeAll("}}");
    }

    if (options.max_output_tokens) |value| {
        try writer.print(",\"max_output_tokens\":{d}", .{value});
    }

    try writer.writeAll(",\"stream\":true,\"store\":false}");
    return try out.toOwnedSlice();
}

/// Builds a streaming Chat Completions request body (used for the xAI native
/// route). The caller owns the returned bytes.
pub fn buildChatCompletionsRequestBody(
    alloc: Allocator,
    model: []const u8,
    serialized_tools: []const u8,
    messages: []const types.ChatMessage,
    options: BuildOptions,
) ![]u8 {
    const verified_index = try verifiedImageIndex(messages, options.verified_images);

    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    const writer = &out.writer;

    try writer.writeAll("{\"model\":");
    try std.json.Stringify.value(model, .{}, writer);
    try writer.writeAll(",\"stream\":true,\"stream_options\":{\"include_usage\":true},\"messages\":[");

    for (messages, 0..) |message, i| {
        if (i > 0) try writer.writeByte(',');
        switch (message.role) {
            .system => {
                try writer.writeAll("{\"role\":\"system\",\"content\":");
                try std.json.Stringify.value(message.content orelse "", .{}, writer);
                try writer.writeByte('}');
            },
            .user => try writeChatCompletionsUserMessage(
                alloc,
                writer,
                message,
                if (verified_index == i) options.verified_images else null,
            ),
            .assistant => {
                try writer.writeAll("{\"role\":\"assistant\"");
                if (message.content) |content| {
                    try writer.writeAll(",\"content\":");
                    try std.json.Stringify.value(content, .{}, writer);
                }
                if (message.tool_calls.len > 0) {
                    try writer.writeAll(",\"tool_calls\":[");
                    for (message.tool_calls, 0..) |call, call_index| {
                        if (call_index > 0) try writer.writeByte(',');
                        try writer.writeAll("{\"id\":");
                        try std.json.Stringify.value(call.id, .{}, writer);
                        try writer.writeAll(",\"type\":\"function\",\"function\":{\"name\":");
                        try std.json.Stringify.value(call.name, .{}, writer);
                        try writer.writeAll(",\"arguments\":");
                        try std.json.Stringify.value(call.arguments_json, .{}, writer);
                        try writer.writeAll("}}");
                    }
                    try writer.writeByte(']');
                }
                try writer.writeByte('}');
            },
            .tool => {
                try writer.writeAll("{\"role\":\"tool\",\"tool_call_id\":");
                try std.json.Stringify.value(message.tool_call_id orelse "", .{}, writer);
                try writer.writeAll(",\"content\":");
                try std.json.Stringify.value(message.content orelse "", .{}, writer);
                try writer.writeByte('}');
            },
        }
    }
    try writer.writeByte(']');

    try writer.writeAll(",\"tools\":");
    try writeChatCompletionsTools(alloc, writer, serialized_tools);

    try writer.writeAll(",\"tool_choice\":");
    if (options.required_tool_name) |name| {
        if (name.len == 0) return error.InvalidRequiredToolName;
        try writer.writeAll("{\"type\":\"function\",\"function\":{\"name\":");
        try std.json.Stringify.value(name, .{}, writer);
        try writer.writeAll("}}");
    } else {
        try std.json.Stringify.value(options.tool_choice.label(), .{}, writer);
    }

    if (options.provider_options.parallel_tool_calls) |parallel| {
        try writer.writeAll(",\"parallel_tool_calls\":");
        try writer.writeAll(if (parallel) "true" else "false");
    }

    if (options.provider_options.reasoning) |*reasoning| {
        if (namedReasoningEffort(reasoning)) |effort| {
            try writer.writeAll(",\"reasoning_effort\":");
            try std.json.Stringify.value(effort, .{}, writer);
        }
    }

    if (options.response_format) |format| {
        var schema = try parseStructuredSchema(alloc, format.schema_json);
        defer schema.deinit();
        try writer.writeAll(",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":");
        try std.json.Stringify.value(format.name, .{}, writer);
        if (format.description.len > 0) {
            try writer.writeAll(",\"description\":");
            try std.json.Stringify.value(format.description, .{}, writer);
        }
        try writer.writeAll(",\"strict\":false,\"schema\":");
        try std.json.Stringify.value(schema.value, .{}, writer);
        try writer.writeAll("}}");
    }

    if (options.max_output_tokens) |value| {
        try writer.print(",\"max_tokens\":{d}", .{value});
    }

    try writer.writeByte('}');
    return try out.toOwnedSlice();
}

// ---------------------------------------------------------------------------
// Message content, images, and structured output
// ---------------------------------------------------------------------------

/// Verified snapshots replace the inline attachments of the final user message,
/// matching `gateway_json.buildGatewayRequestBodyWithVerifiedImagesAndBudget`.
fn verifiedImageIndex(
    messages: []const types.ChatMessage,
    verified_images: ?[]const image_attachments.VerifiedSnapshot,
) error{InvalidRequestHistory}!?usize {
    if (verified_images == null) return null;
    if (messages.len == 0) return error.InvalidRequestHistory;
    const last = messages[messages.len - 1];
    if (last.role != .user or last.images.len != 0) return error.InvalidRequestHistory;
    return messages.len - 1;
}

fn messageHasImages(
    message: types.ChatMessage,
    verified_images: ?[]const image_attachments.VerifiedSnapshot,
) bool {
    if (verified_images) |snapshots| return snapshots.len > 0;
    return message.images.len > 0;
}

/// Media types come from the image sniffing table, so anything outside the
/// token alphabet means the attachment was tampered with.
fn validateMediaType(media_type: []const u8) error{UnsupportedImageMediaType}!void {
    if (media_type.len == 0 or media_type.len > 128) return error.UnsupportedImageMediaType;
    for (media_type) |byte| switch (byte) {
        'a'...'z', 'A'...'Z', '0'...'9', '/', '-', '+', '.' => {},
        else => return error.UnsupportedImageMediaType,
    };
}

/// Writes a complete JSON string holding a `data:` URL. Base64 is emitted in
/// 3-byte-aligned chunks so the encoder output concatenates cleanly.
fn writeImageDataUrl(writer: *std.Io.Writer, media_type: []const u8, bytes: []const u8) !void {
    try validateMediaType(media_type);
    try writer.writeAll("\"data:");
    try writer.writeAll(media_type);
    try writer.writeAll(";base64,");
    var offset: usize = 0;
    while (offset < bytes.len) {
        const end = @min(offset + 3 * 1024, bytes.len);
        try std.base64.standard.Encoder.encodeWriter(writer, bytes[offset..end]);
        offset = end;
    }
    try writer.writeByte('"');
}

const ImagePartWire = enum { responses, chat_completions };

fn writeImagePart(
    writer: *std.Io.Writer,
    wire: ImagePartWire,
    media_type: []const u8,
    bytes: []const u8,
) !void {
    switch (wire) {
        .responses => {
            try writer.writeAll("{\"type\":\"input_image\",\"image_url\":");
            try writeImageDataUrl(writer, media_type, bytes);
            try writer.writeByte('}');
        },
        .chat_completions => {
            try writer.writeAll("{\"type\":\"image_url\",\"image_url\":{\"url\":");
            try writeImageDataUrl(writer, media_type, bytes);
            try writer.writeAll("}}");
        },
    }
}

/// Emits the image parts of one user message, either from pre-verified
/// snapshots or by loading and verifying each inline attachment.
fn writeMessageImageParts(
    alloc: Allocator,
    writer: *std.Io.Writer,
    wire: ImagePartWire,
    message: types.ChatMessage,
    verified_images: ?[]const image_attachments.VerifiedSnapshot,
    wrote_part: *bool,
) !void {
    if (verified_images) |snapshots| {
        for (snapshots) |snapshot| {
            if (wrote_part.*) try writer.writeByte(',');
            try writeImagePart(writer, wire, snapshot.media_type, snapshot.bytes);
            wrote_part.* = true;
        }
        return;
    }
    for (message.images) |attachment| {
        var snapshot = try image_attachments.loadVerifiedSnapshot(alloc, attachment, .{});
        defer snapshot.deinit(alloc);
        if (wrote_part.*) try writer.writeByte(',');
        try writeImagePart(writer, wire, snapshot.media_type, snapshot.bytes);
        wrote_part.* = true;
    }
}

fn writeResponsesUserMessage(
    alloc: Allocator,
    writer: *std.Io.Writer,
    message: types.ChatMessage,
    verified_images: ?[]const image_attachments.VerifiedSnapshot,
) !void {
    const content = message.content orelse "";
    const has_images = messageHasImages(message, verified_images);

    try writer.writeAll("{\"type\":\"message\",\"role\":\"user\",\"content\":[");
    var wrote_part = false;
    if (content.len > 0 or !has_images) {
        try writer.writeAll("{\"type\":\"input_text\",\"text\":");
        try std.json.Stringify.value(content, .{}, writer);
        try writer.writeByte('}');
        wrote_part = true;
    }
    try writeMessageImageParts(alloc, writer, .responses, message, verified_images, &wrote_part);
    try writer.writeAll("]}");
}

fn writeChatCompletionsUserMessage(
    alloc: Allocator,
    writer: *std.Io.Writer,
    message: types.ChatMessage,
    verified_images: ?[]const image_attachments.VerifiedSnapshot,
) !void {
    const content = message.content orelse "";
    if (!messageHasImages(message, verified_images)) {
        try writer.writeAll("{\"role\":\"user\",\"content\":");
        try std.json.Stringify.value(content, .{}, writer);
        try writer.writeByte('}');
        return;
    }

    try writer.writeAll("{\"role\":\"user\",\"content\":[");
    var wrote_part = false;
    if (content.len > 0) {
        try writer.writeAll("{\"type\":\"text\",\"text\":");
        try std.json.Stringify.value(content, .{}, writer);
        try writer.writeByte('}');
        wrote_part = true;
    }
    try writeMessageImageParts(alloc, writer, .chat_completions, message, verified_images, &wrote_part);
    try writer.writeAll("]}");
}

fn parseStructuredSchema(alloc: Allocator, schema_json: []const u8) !std.json.Parsed(std.json.Value) {
    var parsed = std.json.parseFromSlice(std.json.Value, alloc, schema_json, .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidStructuredResponseSchema,
    };
    if (parsed.value != .object) {
        parsed.deinit();
        return error.InvalidStructuredResponseSchema;
    }
    return parsed;
}

/// Named efforts reach the wire; `auto` and "no effort selected" stay silent so
/// the provider keeps its own default. The returned bytes borrow `reasoning`,
/// so the caller must keep it alive while writing.
fn namedReasoningEffort(reasoning: *const types.ReasoningEffort) ?[]const u8 {
    return reasoning.gatewayValue();
}

fn joinSystemInstructions(alloc: Allocator, messages: []const types.ChatMessage) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();

    var wrote = false;
    for (messages) |message| {
        if (message.role != .system) continue;
        const content = message.content orelse continue;
        if (content.len == 0) continue;
        if (wrote) try out.writer.writeAll("\n\n");
        try out.writer.writeAll(content);
        wrote = true;
    }
    return try out.toOwnedSlice();
}

const ToolSpec = struct {
    name: []const u8,
    description: []const u8,
    parameters: ?std.json.Value,
};

fn parseToolsArray(alloc: Allocator, serialized_tools: []const u8) !std.json.Parsed(std.json.Value) {
    const trimmed = std.mem.trim(u8, serialized_tools, " \t\r\n");
    const source = if (trimmed.len == 0) "[]" else trimmed;
    var parsed = std.json.parseFromSlice(std.json.Value, alloc, source, .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidToolSchema,
    };
    if (parsed.value != .array) {
        parsed.deinit();
        return error.InvalidToolSchema;
    }
    return parsed;
}

fn toolSpec(item: std.json.Value) !ToolSpec {
    if (item != .object) return error.InvalidToolSchema;
    const name_value = item.object.get("name") orelse return error.InvalidToolSchema;
    if (name_value != .string or name_value.string.len == 0) return error.InvalidToolSchema;

    const description = if (item.object.get("description")) |value|
        (if (value == .string) value.string else "")
    else
        "";
    const parameters: ?std.json.Value = if (item.object.get("inputSchema")) |value|
        (if (value == .object) value else null)
    else
        null;

    return .{ .name = name_value.string, .description = description, .parameters = parameters };
}

fn writeToolParameters(writer: *std.Io.Writer, parameters: ?std.json.Value) !void {
    if (parameters) |value| {
        try std.json.Stringify.value(value, .{}, writer);
    } else {
        try writer.writeAll("{\"type\":\"object\",\"properties\":{}}");
    }
}

fn writeResponsesTools(alloc: Allocator, writer: *std.Io.Writer, serialized_tools: []const u8) !void {
    var parsed = try parseToolsArray(alloc, serialized_tools);
    defer parsed.deinit();

    try writer.writeByte('[');
    for (parsed.value.array.items, 0..) |item, i| {
        const spec = try toolSpec(item);
        if (i > 0) try writer.writeByte(',');
        try writer.writeAll("{\"type\":\"function\",\"name\":");
        try std.json.Stringify.value(spec.name, .{}, writer);
        try writer.writeAll(",\"description\":");
        try std.json.Stringify.value(spec.description, .{}, writer);
        try writer.writeAll(",\"parameters\":");
        try writeToolParameters(writer, spec.parameters);
        try writer.writeAll(",\"strict\":false}");
    }
    try writer.writeByte(']');
}

fn writeChatCompletionsTools(alloc: Allocator, writer: *std.Io.Writer, serialized_tools: []const u8) !void {
    var parsed = try parseToolsArray(alloc, serialized_tools);
    defer parsed.deinit();

    try writer.writeByte('[');
    for (parsed.value.array.items, 0..) |item, i| {
        const spec = try toolSpec(item);
        if (i > 0) try writer.writeByte(',');
        try writer.writeAll("{\"type\":\"function\",\"function\":{\"name\":");
        try std.json.Stringify.value(spec.name, .{}, writer);
        try writer.writeAll(",\"description\":");
        try std.json.Stringify.value(spec.description, .{}, writer);
        try writer.writeAll(",\"parameters\":");
        try writeToolParameters(writer, spec.parameters);
        try writer.writeAll("}}");
    }
    try writer.writeByte(']');
}

// ---------------------------------------------------------------------------
// SSE framing
// ---------------------------------------------------------------------------

const LineRead = union(enum) {
    line: []const u8,
    read_failed,
    eof,
};

const EventRead = union(enum) {
    data: []const u8,
    done,
    ignored,
    read_failed,
    eof,
};

/// Minimal SSE line framer over a transport-owned reader. It mirrors the
/// gateway framer in `client.zig`, which is private to that file.
const EventReader = struct {
    pending_line: std.ArrayList(u8) = .empty,
    max_line_bytes: usize = max_sse_event_line_bytes,

    fn deinit(self: *EventReader, alloc: Allocator) void {
        self.pending_line.deinit(alloc);
    }

    fn releaseLine(self: *EventReader) void {
        self.pending_line.clearRetainingCapacity();
    }

    fn next(self: *EventReader, alloc: Allocator, reader: *std.Io.Reader) !EventRead {
        const line = switch (try self.readLine(alloc, reader)) {
            .line => |line| line,
            .read_failed => return .read_failed,
            .eof => return .eof,
        };

        const trimmed = std.mem.trimEnd(u8, line, "\r");
        if (trimmed.len == 0) return .ignored;
        if (trimmed[0] == ':') return .ignored;

        const data_prefix = "data:";
        if (!std.mem.startsWith(u8, trimmed, data_prefix)) return .ignored;

        var payload = trimmed[data_prefix.len..];
        if (payload.len > 0 and payload[0] == ' ') payload = payload[1..];
        if (std.mem.eql(u8, payload, "[DONE]")) return .done;
        if (payload.len == 0) return .ignored;
        return .{ .data = payload };
    }

    fn readLine(self: *EventReader, alloc: Allocator, reader: *std.Io.Reader) !LineRead {
        while (true) {
            const fragment = reader.takeDelimiter('\n') catch |err| switch (err) {
                error.StreamTooLong => {
                    const buffered = reader.buffered();
                    if (buffered.len == 0) return error.OpenAiSseReadStalled;
                    if (buffered.len > self.max_line_bytes - self.pending_line.items.len) {
                        return error.OpenAiSseEventTooLarge;
                    }
                    try self.pending_line.appendSlice(alloc, buffered);
                    reader.tossBuffered();
                    continue;
                },
                error.ReadFailed => return .read_failed,
            } orelse {
                if (self.pending_line.items.len > 0) return .{ .line = self.pending_line.items };
                return .eof;
            };

            if (fragment.len > self.max_line_bytes - self.pending_line.items.len) {
                return error.OpenAiSseEventTooLarge;
            }
            if (self.pending_line.items.len == 0) return .{ .line = fragment };
            try self.pending_line.appendSlice(alloc, fragment);
            return .{ .line = self.pending_line.items };
        }
    }
};

// ---------------------------------------------------------------------------
// Shared stream accumulation
// ---------------------------------------------------------------------------

const ToolAccumulator = struct {
    /// Responses API item identity (`item.id`); empty on the chat wire.
    item_id: std.ArrayList(u8) = .empty,
    /// Responses API `output_index`; null on the chat wire.
    output_index: ?i64 = null,
    /// Chat Completions `delta.tool_calls[].index`; null on the responses wire.
    delta_index: ?i64 = null,
    id: std.ArrayList(u8) = .empty,
    name: std.ArrayList(u8) = .empty,
    arguments: std.ArrayList(u8) = .empty,
    started: bool = false,
    done: bool = false,

    fn deinit(self: *ToolAccumulator, alloc: Allocator) void {
        self.item_id.deinit(alloc);
        self.id.deinit(alloc);
        self.name.deinit(alloc);
        self.arguments.deinit(alloc);
    }
};

const ToolAccumulatorList = std.ArrayList(ToolAccumulator);

fn deinitAccumulators(alloc: Allocator, list: *ToolAccumulatorList) void {
    for (list.items) |*item| item.deinit(alloc);
    list.deinit(alloc);
}

fn materializeToolCalls(alloc: Allocator, accumulators: []const ToolAccumulator) ![]types.ToolCall {
    if (accumulators.len == 0) return &.{};

    const calls = try alloc.alloc(types.ToolCall, accumulators.len);
    var built: usize = 0;
    errdefer {
        for (calls[0..built]) |call| {
            alloc.free(@constCast(call.id));
            alloc.free(@constCast(call.name));
            alloc.free(@constCast(call.arguments_json));
        }
        alloc.free(calls);
    }

    for (accumulators, 0..) |accumulator, i| {
        const id = try alloc.dupe(u8, accumulator.id.items);
        errdefer alloc.free(id);
        const name = try alloc.dupe(u8, accumulator.name.items);
        errdefer alloc.free(name);
        const source = if (accumulator.arguments.items.len == 0) "{}" else accumulator.arguments.items;
        const arguments = try alloc.dupe(u8, source);
        errdefer alloc.free(arguments);

        calls[i] = .{
            .id = id,
            .name = name,
            .arguments_json = arguments,
            .argument_integrity = try types.ToolArgumentIntegrity.classifySerialized(alloc, arguments),
        };
        built += 1;
    }
    return calls;
}

fn freeCompletion(alloc: Allocator, completion: *types.GatewayCompletion) void {
    if (completion.content) |content| alloc.free(@constCast(content));
    if (completion.generation_id) |id| alloc.free(@constCast(id));
    if (completion.provider_failure_detail) |detail| alloc.free(@constCast(detail));
    types.freeToolCallSlice(alloc, @constCast(completion.tool_calls));
    completion.* = .{};
}

/// Appends a content delta under the capture limit while the callback still
/// receives the full delta, exactly like the gateway SSE consumer.
fn captureContent(
    alloc: Allocator,
    buffer: *std.ArrayList(u8),
    limit: ?usize,
    delta: []const u8,
) !void {
    const retained = if (limit) |max|
        delta[0..@min(delta.len, max -| buffer.items.len)]
    else
        delta;
    try buffer.appendSlice(alloc, retained);
}

fn objString(object: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const value = object.get(key) orelse return null;
    return if (value == .string) value.string else null;
}

fn objInteger(object: std.json.ObjectMap, key: []const u8) ?i64 {
    const value = object.get(key) orelse return null;
    return if (value == .integer) value.integer else null;
}

fn objObject(object: std.json.ObjectMap, key: []const u8) ?std.json.ObjectMap {
    const value = object.get(key) orelse return null;
    return if (value == .object) value.object else null;
}

fn tokenCount(value: ?i64) ?u64 {
    const raw = value orelse return null;
    if (raw < 0) return null;
    return @intCast(raw);
}

/// Extracts a human-readable provider failure message from an event payload.
fn failureMessageText(value: std.json.Value) ?[]const u8 {
    switch (value) {
        .string => |text| return if (text.len > 0) text else null,
        .object => |object| {
            if (objString(object, "message")) |text| {
                if (text.len > 0) return text;
            }
            if (object.get("error")) |nested| {
                if (failureMessageText(nested)) |text| return text;
            }
            if (object.get("response")) |nested| {
                if (failureMessageText(nested)) |text| return text;
            }
            if (object.get("incomplete_details")) |nested| {
                if (failureMessageText(nested)) |text| return text;
            }
            if (objString(object, "reason")) |text| {
                if (text.len > 0) return text;
            }
            if (objString(object, "code")) |text| {
                if (text.len > 0) return text;
            }
            return null;
        },
        else => return null,
    }
}

fn captureFailureDetail(alloc: Allocator, current: *?[]u8, root: std.json.Value) !void {
    if (current.* != null) return;
    const text = failureMessageText(root) orelse return;
    current.* = try alloc.dupe(u8, text);
}

fn matchAccumulator(
    accumulators: []const ToolAccumulator,
    item_id: ?[]const u8,
    call_id: ?[]const u8,
    output_index: ?i64,
) ?usize {
    if (item_id) |needle| {
        if (needle.len > 0) {
            for (accumulators, 0..) |accumulator, i| {
                if (accumulator.item_id.items.len > 0 and
                    std.mem.eql(u8, accumulator.item_id.items, needle)) return i;
            }
        }
    }
    if (call_id) |needle| {
        if (needle.len > 0) {
            for (accumulators, 0..) |accumulator, i| {
                if (accumulator.id.items.len > 0 and
                    std.mem.eql(u8, accumulator.id.items, needle)) return i;
            }
        }
    }
    if (output_index) |needle| {
        for (accumulators, 0..) |accumulator, i| {
            if (accumulator.output_index) |candidate| {
                if (candidate == needle) return i;
            }
        }
    }
    return null;
}

fn lastOpenAccumulator(accumulators: []const ToolAccumulator) ?usize {
    var i = accumulators.len;
    while (i > 0) {
        i -= 1;
        if (!accumulators[i].done) return i;
    }
    return null;
}

// ---------------------------------------------------------------------------
// Responses API stream
// ---------------------------------------------------------------------------

/// Decodes an OpenAI Responses API SSE stream. Every populated field of the
/// returned completion is owned by `alloc`.
pub fn consumeResponsesSseStream(
    alloc: Allocator,
    reader: *std.Io.Reader,
    callback_ctx: *anyopaque,
    on_content_chunk: agent_stream_provider.StreamCallback,
    on_tool_start: ?agent_stream_provider.ToolStartCallback,
    on_reasoning_chunk: ?agent_stream_provider.StreamCallback,
    on_tool_input_chunk: ?agent_stream_provider.StreamCallback,
    cancel_flag: *std.atomic.Value(bool),
    content_capture_limit: ?usize,
) anyerror!types.GatewayCompletion {
    var content_buf: std.ArrayList(u8) = .empty;
    defer content_buf.deinit(alloc);

    var accumulators: ToolAccumulatorList = .empty;
    defer deinitAccumulators(alloc, &accumulators);

    var generation_id: ?[]u8 = null;
    errdefer if (generation_id) |id| alloc.free(id);
    var provider_failure_detail: ?[]u8 = null;
    errdefer if (provider_failure_detail) |detail| alloc.free(detail);

    var finish_reason: ?types.ProviderFinishReason = null;
    var usage: types.Usage = .{};

    var event_reader = EventReader{};
    defer event_reader.deinit(alloc);

    stream: while (true) {
        if (cancel_flag.load(.seq_cst)) return error.Cancelled;

        const event = try event_reader.next(alloc, reader);
        defer event_reader.releaseLine();

        const json_text = switch (event) {
            .data => |text| text,
            .done, .eof => break :stream,
            .ignored => continue,
            .read_failed => {
                if (cancel_flag.load(.seq_cst)) return error.Cancelled;
                return error.ReadFailed;
            },
        };

        var parsed = std.json.parseFromSlice(std.json.Value, alloc, json_text, .{}) catch |err| {
            if (err == error.OutOfMemory) return err;
            continue;
        };
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) continue;
        const event_type = objString(root.object, "type") orelse continue;

        if (std.mem.eql(u8, event_type, "response.created")) {
            if (generation_id == null) {
                if (objObject(root.object, "response")) |response| {
                    if (objString(response, "id")) |id| {
                        if (id.len > 0) generation_id = try alloc.dupe(u8, id);
                    }
                }
            }
        } else if (std.mem.eql(u8, event_type, "response.output_text.delta")) {
            if (objString(root.object, "delta")) |delta| {
                if (delta.len > 0) {
                    try captureContent(alloc, &content_buf, content_capture_limit, delta);
                    on_content_chunk(callback_ctx, delta);
                }
            }
        } else if (std.mem.eql(u8, event_type, "response.reasoning_summary_text.delta") or
            std.mem.eql(u8, event_type, "response.reasoning_text.delta"))
        {
            if (on_reasoning_chunk) |callback| {
                if (objString(root.object, "delta")) |delta| {
                    if (delta.len > 0) callback(callback_ctx, delta);
                }
            }
        } else if (std.mem.eql(u8, event_type, "response.output_item.added")) {
            const item = objObject(root.object, "item") orelse continue;
            const item_type = objString(item, "type") orelse continue;
            if (!std.mem.eql(u8, item_type, "function_call")) continue;

            const call_id = objString(item, "call_id") orelse "";
            const name = objString(item, "name") orelse "";
            if (matchAccumulator(
                accumulators.items,
                objString(item, "id"),
                if (call_id.len > 0) call_id else null,
                objInteger(root.object, "output_index"),
            ) != null) continue;

            var accumulator = ToolAccumulator{
                .output_index = objInteger(root.object, "output_index"),
            };
            errdefer accumulator.deinit(alloc);
            if (objString(item, "id")) |item_id| try accumulator.item_id.appendSlice(alloc, item_id);
            try accumulator.id.appendSlice(alloc, call_id);
            try accumulator.name.appendSlice(alloc, name);
            accumulator.started = true;
            try accumulators.append(alloc, accumulator);

            if (on_tool_start) |callback| {
                if (call_id.len > 0 and name.len > 0) callback(callback_ctx, call_id, name, null);
            }
        } else if (std.mem.eql(u8, event_type, "response.function_call_arguments.delta")) {
            const delta = objString(root.object, "delta") orelse continue;
            if (delta.len == 0) continue;
            const index = matchAccumulator(
                accumulators.items,
                objString(root.object, "item_id"),
                null,
                objInteger(root.object, "output_index"),
            ) orelse lastOpenAccumulator(accumulators.items) orelse continue;
            try accumulators.items[index].arguments.appendSlice(alloc, delta);
            if (on_tool_input_chunk) |callback| callback(callback_ctx, delta);
        } else if (std.mem.eql(u8, event_type, "response.output_item.done")) {
            const item = objObject(root.object, "item") orelse continue;
            const item_type = objString(item, "type") orelse continue;
            if (!std.mem.eql(u8, item_type, "function_call")) continue;

            const call_id = objString(item, "call_id") orelse "";
            const index = matchAccumulator(
                accumulators.items,
                objString(item, "id"),
                if (call_id.len > 0) call_id else null,
                objInteger(root.object, "output_index"),
            ) orelse blk: {
                var accumulator = ToolAccumulator{
                    .output_index = objInteger(root.object, "output_index"),
                };
                errdefer accumulator.deinit(alloc);
                if (objString(item, "id")) |item_id| try accumulator.item_id.appendSlice(alloc, item_id);
                try accumulator.id.appendSlice(alloc, call_id);
                try accumulator.name.appendSlice(alloc, objString(item, "name") orelse "");
                try accumulators.append(alloc, accumulator);
                break :blk accumulators.items.len - 1;
            };

            const accumulator = &accumulators.items[index];
            if (accumulator.id.items.len == 0 and call_id.len > 0) {
                try accumulator.id.appendSlice(alloc, call_id);
            }
            if (accumulator.name.items.len == 0) {
                if (objString(item, "name")) |name| try accumulator.name.appendSlice(alloc, name);
            }
            if (objString(item, "arguments")) |arguments| {
                accumulator.arguments.clearRetainingCapacity();
                try accumulator.arguments.appendSlice(alloc, arguments);
            }
            accumulator.done = true;
        } else if (std.mem.eql(u8, event_type, "response.completed")) {
            if (objObject(root.object, "response")) |response| {
                if (objObject(response, "usage")) |usage_object| {
                    usage.input_tokens = tokenCount(objInteger(usage_object, "input_tokens"));
                    usage.output_tokens = tokenCount(objInteger(usage_object, "output_tokens"));
                }
            }
            finish_reason = if (accumulators.items.len > 0) .tool_calls else .stop;
        } else if (std.mem.eql(u8, event_type, "response.failed")) {
            try captureFailureDetail(alloc, &provider_failure_detail, root);
            finish_reason = .provider_error;
        } else if (std.mem.eql(u8, event_type, "response.incomplete")) {
            try captureFailureDetail(alloc, &provider_failure_detail, root);
            var reason: ?[]const u8 = null;
            if (objObject(root.object, "response")) |response| {
                if (objObject(response, "incomplete_details")) |details| {
                    reason = objString(details, "reason");
                }
            }
            finish_reason = if (reason != null and std.mem.eql(u8, reason.?, "max_output_tokens"))
                .length
            else
                .other;
        } else if (std.mem.eql(u8, event_type, "error")) {
            try captureFailureDetail(alloc, &provider_failure_detail, root);
        }
    }

    var completion: types.GatewayCompletion = .{};
    errdefer freeCompletion(alloc, &completion);

    if (content_buf.items.len > 0) completion.content = try alloc.dupe(u8, content_buf.items);
    completion.tool_calls = try materializeToolCalls(alloc, accumulators.items);
    completion.generation_id = generation_id;
    generation_id = null;
    completion.provider_failure_detail = provider_failure_detail;
    provider_failure_detail = null;
    completion.finish_reason = finish_reason;
    completion.usage = usage;
    return completion;
}

// ---------------------------------------------------------------------------
// Chat Completions stream
// ---------------------------------------------------------------------------

fn parseChatFinishReason(raw: []const u8) types.ProviderFinishReason {
    if (std.mem.eql(u8, raw, "stop")) return .stop;
    if (std.mem.eql(u8, raw, "length")) return .length;
    if (std.mem.eql(u8, raw, "tool_calls")) return .tool_calls;
    if (std.mem.eql(u8, raw, "content_filter")) return .content_filter;
    return .other;
}

/// Decodes an OpenAI-compatible Chat Completions SSE stream. Every populated
/// field of the returned completion is owned by `alloc`.
pub fn consumeChatCompletionsSseStream(
    alloc: Allocator,
    reader: *std.Io.Reader,
    callback_ctx: *anyopaque,
    on_content_chunk: agent_stream_provider.StreamCallback,
    on_tool_start: ?agent_stream_provider.ToolStartCallback,
    on_reasoning_chunk: ?agent_stream_provider.StreamCallback,
    on_tool_input_chunk: ?agent_stream_provider.StreamCallback,
    cancel_flag: *std.atomic.Value(bool),
    content_capture_limit: ?usize,
) anyerror!types.GatewayCompletion {
    var content_buf: std.ArrayList(u8) = .empty;
    defer content_buf.deinit(alloc);

    var accumulators: ToolAccumulatorList = .empty;
    defer deinitAccumulators(alloc, &accumulators);

    var generation_id: ?[]u8 = null;
    errdefer if (generation_id) |id| alloc.free(id);
    var provider_failure_detail: ?[]u8 = null;
    errdefer if (provider_failure_detail) |detail| alloc.free(detail);

    var finish_reason: ?types.ProviderFinishReason = null;
    var usage: types.Usage = .{};

    var event_reader = EventReader{};
    defer event_reader.deinit(alloc);

    stream: while (true) {
        if (cancel_flag.load(.seq_cst)) return error.Cancelled;

        const event = try event_reader.next(alloc, reader);
        defer event_reader.releaseLine();

        const json_text = switch (event) {
            .data => |text| text,
            .done, .eof => break :stream,
            .ignored => continue,
            .read_failed => {
                if (cancel_flag.load(.seq_cst)) return error.Cancelled;
                return error.ReadFailed;
            },
        };

        var parsed = std.json.parseFromSlice(std.json.Value, alloc, json_text, .{}) catch |err| {
            if (err == error.OutOfMemory) return err;
            continue;
        };
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) continue;

        if (generation_id == null) {
            if (objString(root.object, "id")) |id| {
                if (id.len > 0) generation_id = try alloc.dupe(u8, id);
            }
        }

        if (objObject(root.object, "usage")) |usage_object| {
            if (tokenCount(objInteger(usage_object, "prompt_tokens"))) |value| {
                usage.input_tokens = value;
            }
            if (tokenCount(objInteger(usage_object, "completion_tokens"))) |value| {
                usage.output_tokens = value;
            }
        }

        if (root.object.get("error")) |error_value| {
            try captureFailureDetail(alloc, &provider_failure_detail, error_value);
        }

        const choices = root.object.get("choices") orelse continue;
        if (choices != .array or choices.array.items.len == 0) continue;
        const choice = choices.array.items[0];
        if (choice != .object) continue;

        if (objString(choice.object, "finish_reason")) |raw| {
            if (raw.len > 0) finish_reason = parseChatFinishReason(raw);
        }

        const delta = objObject(choice.object, "delta") orelse continue;

        if (objString(delta, "content")) |text| {
            if (text.len > 0) {
                try captureContent(alloc, &content_buf, content_capture_limit, text);
                on_content_chunk(callback_ctx, text);
            }
        }

        if (on_reasoning_chunk) |callback| {
            if (objString(delta, "reasoning_content")) |text| {
                if (text.len > 0) callback(callback_ctx, text);
            }
        }

        const tool_calls = delta.get("tool_calls") orelse continue;
        if (tool_calls != .array) continue;
        for (tool_calls.array.items) |entry| {
            if (entry != .object) continue;
            const delta_index = objInteger(entry.object, "index") orelse 0;

            var found: ?usize = null;
            for (accumulators.items, 0..) |accumulator, i| {
                if (accumulator.delta_index) |candidate| {
                    if (candidate == delta_index) {
                        found = i;
                        break;
                    }
                }
            }
            const index = found orelse blk: {
                var accumulator = ToolAccumulator{ .delta_index = delta_index };
                errdefer accumulator.deinit(alloc);
                try accumulators.append(alloc, accumulator);
                break :blk accumulators.items.len - 1;
            };

            const accumulator = &accumulators.items[index];
            if (objString(entry.object, "id")) |id| {
                if (id.len > 0 and accumulator.id.items.len == 0) {
                    try accumulator.id.appendSlice(alloc, id);
                }
            }

            if (objObject(entry.object, "function")) |function| {
                if (objString(function, "name")) |name| {
                    if (name.len > 0 and accumulator.name.items.len == 0) {
                        try accumulator.name.appendSlice(alloc, name);
                    }
                }
                if (objString(function, "arguments")) |arguments| {
                    if (arguments.len > 0) {
                        try accumulator.arguments.appendSlice(alloc, arguments);
                        if (on_tool_input_chunk) |callback| callback(callback_ctx, arguments);
                    }
                }
            }

            if (!accumulator.started and
                accumulator.id.items.len > 0 and
                accumulator.name.items.len > 0)
            {
                accumulator.started = true;
                if (on_tool_start) |callback| {
                    callback(callback_ctx, accumulator.id.items, accumulator.name.items, null);
                }
            }
        }
    }

    var completion: types.GatewayCompletion = .{};
    errdefer freeCompletion(alloc, &completion);

    if (content_buf.items.len > 0) completion.content = try alloc.dupe(u8, content_buf.items);
    completion.tool_calls = try materializeToolCalls(alloc, accumulators.items);
    completion.generation_id = generation_id;
    generation_id = null;
    completion.provider_failure_detail = provider_failure_detail;
    provider_failure_detail = null;
    completion.finish_reason = finish_reason;
    completion.usage = usage;
    return completion;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

const sample_tools =
    \\[{"name":"read_file","description":"Read a file","inputSchema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}]
;

fn sampleConversation() [4]types.ChatMessage {
    const calls = struct {
        const value = [_]types.ToolCall{.{
            .id = "call_1",
            .name = "read_file",
            .arguments_json = "{\"path\":\"src/main.zig\"}",
        }};
    };
    return .{
        .{ .role = .system, .content = "You are fx." },
        .{ .role = .user, .content = "Read main." },
        .{ .role = .assistant, .content = "Reading it.", .tool_calls = &calls.value },
        .{
            .role = .tool,
            .content = "const std = @import(\"std\");",
            .tool_call_id = "call_1",
            .tool_name = "read_file",
        },
    };
}

test "responses request body maps every message role onto the input array" {
    const alloc = testing.allocator;
    const messages = sampleConversation();

    const body = try buildResponsesRequestBody(
        alloc,
        "gpt-5.2-codex",
        sample_tools,
        &messages,
        .{},
    );
    defer alloc.free(body);

    var parsed = try std.json.parseFromSlice(std.json.Value, alloc, body, .{});
    defer parsed.deinit();
    const root = parsed.value.object;

    try testing.expectEqualStrings("gpt-5.2-codex", root.get("model").?.string);
    try testing.expectEqualStrings("You are fx.", root.get("instructions").?.string);
    try testing.expect(root.get("stream").?.bool);
    try testing.expect(!root.get("store").?.bool);
    try testing.expectEqualStrings("auto", root.get("tool_choice").?.string);
    try testing.expect(root.get("reasoning") == null);
    try testing.expect(root.get("parallel_tool_calls") == null);
    try testing.expect(root.get("max_output_tokens") == null);

    const input = root.get("input").?.array.items;
    try testing.expectEqual(@as(usize, 4), input.len);

    try testing.expectEqualStrings("message", input[0].object.get("type").?.string);
    try testing.expectEqualStrings("user", input[0].object.get("role").?.string);
    const user_part = input[0].object.get("content").?.array.items[0].object;
    try testing.expectEqualStrings("input_text", user_part.get("type").?.string);
    try testing.expectEqualStrings("Read main.", user_part.get("text").?.string);

    try testing.expectEqualStrings("assistant", input[1].object.get("role").?.string);
    const assistant_part = input[1].object.get("content").?.array.items[0].object;
    try testing.expectEqualStrings("output_text", assistant_part.get("type").?.string);
    try testing.expectEqualStrings("Reading it.", assistant_part.get("text").?.string);

    try testing.expectEqualStrings("function_call", input[2].object.get("type").?.string);
    try testing.expectEqualStrings("call_1", input[2].object.get("call_id").?.string);
    try testing.expectEqualStrings("read_file", input[2].object.get("name").?.string);
    try testing.expectEqualStrings(
        "{\"path\":\"src/main.zig\"}",
        input[2].object.get("arguments").?.string,
    );

    try testing.expectEqualStrings("function_call_output", input[3].object.get("type").?.string);
    try testing.expectEqualStrings("call_1", input[3].object.get("call_id").?.string);
    try testing.expectEqualStrings(
        "const std = @import(\"std\");",
        input[3].object.get("output").?.string,
    );
}

test "responses request body joins several system messages into instructions" {
    const alloc = testing.allocator;
    const messages = [_]types.ChatMessage{
        .{ .role = .system, .content = "first" },
        .{ .role = .system, .content = "second" },
        .{ .role = .user, .content = "go" },
    };

    const body = try buildResponsesRequestBody(alloc, "gpt-5.2", "[]", &messages, .{ .tool_choice = .none });
    defer alloc.free(body);

    var parsed = try std.json.parseFromSlice(std.json.Value, alloc, body, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("first\n\nsecond", parsed.value.object.get("instructions").?.string);
    try testing.expectEqualStrings("none", parsed.value.object.get("tool_choice").?.string);
    try testing.expectEqual(@as(usize, 1), parsed.value.object.get("input").?.array.items.len);
    try testing.expectEqual(@as(usize, 0), parsed.value.object.get("tools").?.array.items.len);
}

test "chat completions request body maps roles, tool calls, and tool results" {
    const alloc = testing.allocator;
    const messages = sampleConversation();

    const body = try buildChatCompletionsRequestBody(
        alloc,
        "grok-4",
        sample_tools,
        &messages,
        .{},
    );
    defer alloc.free(body);

    var parsed = try std.json.parseFromSlice(std.json.Value, alloc, body, .{});
    defer parsed.deinit();
    const root = parsed.value.object;

    try testing.expectEqualStrings("grok-4", root.get("model").?.string);
    try testing.expect(root.get("stream").?.bool);
    try testing.expect(root.get("stream_options").?.object.get("include_usage").?.bool);
    try testing.expectEqualStrings("auto", root.get("tool_choice").?.string);
    try testing.expect(root.get("max_tokens") == null);
    try testing.expect(root.get("reasoning_effort") == null);

    const wire_messages = root.get("messages").?.array.items;
    try testing.expectEqual(@as(usize, 4), wire_messages.len);
    try testing.expectEqualStrings("system", wire_messages[0].object.get("role").?.string);
    try testing.expectEqualStrings("You are fx.", wire_messages[0].object.get("content").?.string);
    try testing.expectEqualStrings("user", wire_messages[1].object.get("role").?.string);

    const assistant = wire_messages[2].object;
    try testing.expectEqualStrings("assistant", assistant.get("role").?.string);
    try testing.expectEqualStrings("Reading it.", assistant.get("content").?.string);
    const call = assistant.get("tool_calls").?.array.items[0].object;
    try testing.expectEqualStrings("call_1", call.get("id").?.string);
    try testing.expectEqualStrings("function", call.get("type").?.string);
    try testing.expectEqualStrings("read_file", call.get("function").?.object.get("name").?.string);
    try testing.expectEqualStrings(
        "{\"path\":\"src/main.zig\"}",
        call.get("function").?.object.get("arguments").?.string,
    );

    const tool_message = wire_messages[3].object;
    try testing.expectEqualStrings("tool", tool_message.get("role").?.string);
    try testing.expectEqualStrings("call_1", tool_message.get("tool_call_id").?.string);
    try testing.expectEqualStrings(
        "const std = @import(\"std\");",
        tool_message.get("content").?.string,
    );
}

test "chat completions omits assistant content when the message carries none" {
    const alloc = testing.allocator;
    const calls = [_]types.ToolCall{.{ .id = "c1", .name = "list_dir", .arguments_json = "{}" }};
    const messages = [_]types.ChatMessage{
        .{ .role = .user, .content = "list" },
        .{ .role = .assistant, .tool_calls = &calls },
        .{ .role = .tool, .content = "ok", .tool_call_id = "c1", .tool_name = "list_dir" },
    };

    const body = try buildChatCompletionsRequestBody(alloc, "grok-4", "[]", &messages, .{});
    defer alloc.free(body);

    var parsed = try std.json.parseFromSlice(std.json.Value, alloc, body, .{});
    defer parsed.deinit();
    const assistant = parsed.value.object.get("messages").?.array.items[1].object;
    try testing.expect(assistant.get("content") == null);
    try testing.expectEqual(@as(usize, 1), assistant.get("tool_calls").?.array.items.len);
}

test "gateway tool specs transform onto both OpenAI wire shapes" {
    const alloc = testing.allocator;
    const messages = [_]types.ChatMessage{.{ .role = .user, .content = "go" }};

    const responses_body = try buildResponsesRequestBody(
        alloc,
        "gpt-5.2",
        sample_tools,
        &messages,
        .{},
    );
    defer alloc.free(responses_body);

    var responses_parsed = try std.json.parseFromSlice(std.json.Value, alloc, responses_body, .{});
    defer responses_parsed.deinit();
    const responses_tool = responses_parsed.value.object.get("tools").?.array.items[0].object;
    try testing.expectEqualStrings("function", responses_tool.get("type").?.string);
    try testing.expectEqualStrings("read_file", responses_tool.get("name").?.string);
    try testing.expectEqualStrings("Read a file", responses_tool.get("description").?.string);
    try testing.expect(!responses_tool.get("strict").?.bool);
    const responses_parameters = responses_tool.get("parameters").?.object;
    try testing.expectEqualStrings("object", responses_parameters.get("type").?.string);
    try testing.expect(responses_parameters.get("properties").?.object.get("path") != null);
    try testing.expectEqualStrings(
        "path",
        responses_parameters.get("required").?.array.items[0].string,
    );

    const chat_body = try buildChatCompletionsRequestBody(
        alloc,
        "grok-4",
        sample_tools,
        &messages,
        .{},
    );
    defer alloc.free(chat_body);

    var chat_parsed = try std.json.parseFromSlice(std.json.Value, alloc, chat_body, .{});
    defer chat_parsed.deinit();
    const chat_tool = chat_parsed.value.object.get("tools").?.array.items[0].object;
    try testing.expectEqualStrings("function", chat_tool.get("type").?.string);
    const chat_function = chat_tool.get("function").?.object;
    try testing.expectEqualStrings("read_file", chat_function.get("name").?.string);
    try testing.expectEqualStrings("Read a file", chat_function.get("description").?.string);
    try testing.expectEqualStrings(
        "object",
        chat_function.get("parameters").?.object.get("type").?.string,
    );
}

test "provider options reach both wires only when they are set" {
    const alloc = testing.allocator;
    const messages = [_]types.ChatMessage{.{ .role = .user, .content = "go" }};
    const options = model_capabilities.ResolvedProviderOptions{
        .reasoning = types.ReasoningEffort.literal("high"),
        .parallel_tool_calls = false,
    };

    const responses_body = try buildResponsesRequestBody(
        alloc,
        "gpt-5.2",
        "[]",
        &messages,
        .{ .provider_options = options, .max_output_tokens = 4096 },
    );
    defer alloc.free(responses_body);
    var responses_parsed = try std.json.parseFromSlice(std.json.Value, alloc, responses_body, .{});
    defer responses_parsed.deinit();
    try testing.expectEqualStrings(
        "high",
        responses_parsed.value.object.get("reasoning").?.object.get("effort").?.string,
    );
    try testing.expect(!responses_parsed.value.object.get("parallel_tool_calls").?.bool);
    try testing.expectEqual(
        @as(i64, 4096),
        responses_parsed.value.object.get("max_output_tokens").?.integer,
    );

    const chat_body = try buildChatCompletionsRequestBody(
        alloc,
        "grok-4",
        "[]",
        &messages,
        .{ .provider_options = options, .max_output_tokens = 2048 },
    );
    defer alloc.free(chat_body);
    var chat_parsed = try std.json.parseFromSlice(std.json.Value, alloc, chat_body, .{});
    defer chat_parsed.deinit();
    try testing.expectEqualStrings("high", chat_parsed.value.object.get("reasoning_effort").?.string);
    try testing.expect(!chat_parsed.value.object.get("parallel_tool_calls").?.bool);
    try testing.expectEqual(@as(i64, 2048), chat_parsed.value.object.get("max_tokens").?.integer);
}

test "automatic reasoning effort stays off both wires" {
    const alloc = testing.allocator;
    const messages = [_]types.ChatMessage{.{ .role = .user, .content = "go" }};
    const options = model_capabilities.ResolvedProviderOptions{ .reasoning = .auto };

    const responses_body = try buildResponsesRequestBody(alloc, "gpt-5.2", "[]", &messages, .{ .provider_options = options });
    defer alloc.free(responses_body);
    var responses_parsed = try std.json.parseFromSlice(std.json.Value, alloc, responses_body, .{});
    defer responses_parsed.deinit();
    try testing.expect(responses_parsed.value.object.get("reasoning") == null);

    const chat_body = try buildChatCompletionsRequestBody(alloc, "grok-4", "[]", &messages, .{ .provider_options = options });
    defer alloc.free(chat_body);
    var chat_parsed = try std.json.parseFromSlice(std.json.Value, alloc, chat_body, .{});
    defer chat_parsed.deinit();
    try testing.expect(chat_parsed.value.object.get("reasoning_effort") == null);
}

/// The one-pixel PNG header the image sniffer recognizes; its base64 form is
/// asserted in the image tests below.
const png_fixture_bytes = "\x89PNG\r\n\x1a\nabc";
const png_fixture_base64 = "iVBORw0KGgphYmM=";

test "inline message images become data-url parts on both wires" {
    const alloc = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    {
        var file = try tmp.dir.createFile(std.testing.io, "image.png", .{});
        defer file.close(std.testing.io);
        try file.writeStreamingAll(std.testing.io, png_fixture_bytes);
    }

    const image_path = try io_mod.dirRealpathAlloc(alloc, tmp.dir, "image.png");
    defer alloc.free(image_path);
    const source = [_]types.ImageAttachment{.{
        .id = 1,
        .path = @constCast(image_path),
        .media_type = @constCast("image/png"),
    }};
    const images = try types.dupeImageAttachmentSlice(alloc, &source);
    defer types.freeImageAttachmentSlice(alloc, images);

    const root = try io_mod.dirRealpathAlloc(alloc, tmp.dir, ".");
    defer alloc.free(root);
    const snapshot_dir = try std.fs.path.join(alloc, &.{ root, "snapshots" });
    defer alloc.free(snapshot_dir);
    try image_attachments.captureImageSnapshot(alloc, &images[0], snapshot_dir);

    const messages = [_]types.ChatMessage{
        .{ .role = .user, .content = "look", .images = images },
    };
    const expected_url = "data:image/png;base64," ++ png_fixture_base64;

    const responses_body = try buildResponsesRequestBody(alloc, "gpt-5.2", "[]", &messages, .{});
    defer alloc.free(responses_body);
    var responses_parsed = try std.json.parseFromSlice(std.json.Value, alloc, responses_body, .{});
    defer responses_parsed.deinit();
    const responses_parts = responses_parsed.value.object
        .get("input").?.array.items[0].object
        .get("content").?.array.items;
    try testing.expectEqual(@as(usize, 2), responses_parts.len);
    try testing.expectEqualStrings("input_text", responses_parts[0].object.get("type").?.string);
    try testing.expectEqualStrings("look", responses_parts[0].object.get("text").?.string);
    try testing.expectEqualStrings("input_image", responses_parts[1].object.get("type").?.string);
    try testing.expectEqualStrings(expected_url, responses_parts[1].object.get("image_url").?.string);

    const chat_body = try buildChatCompletionsRequestBody(alloc, "grok-4", "[]", &messages, .{});
    defer alloc.free(chat_body);
    var chat_parsed = try std.json.parseFromSlice(std.json.Value, alloc, chat_body, .{});
    defer chat_parsed.deinit();
    const chat_parts = chat_parsed.value.object
        .get("messages").?.array.items[0].object
        .get("content").?.array.items;
    try testing.expectEqual(@as(usize, 2), chat_parts.len);
    try testing.expectEqualStrings("text", chat_parts[0].object.get("type").?.string);
    try testing.expectEqualStrings("look", chat_parts[0].object.get("text").?.string);
    try testing.expectEqualStrings("image_url", chat_parts[1].object.get("type").?.string);
    try testing.expectEqualStrings(
        expected_url,
        chat_parts[1].object.get("image_url").?.object.get("url").?.string,
    );
}

test "verified snapshots attach to the final user message on both wires" {
    const alloc = testing.allocator;
    var bytes = png_fixture_bytes.*;
    const snapshots = [_]image_attachments.VerifiedSnapshot{.{
        .bytes = &bytes,
        .media_type = "image/png",
    }};
    const messages = [_]types.ChatMessage{
        .{ .role = .system, .content = "rules" },
        .{ .role = .user, .content = "inspect" },
    };
    const expected_url = "data:image/png;base64," ++ png_fixture_base64;
    const options = BuildOptions{ .verified_images = &snapshots };

    const responses_body = try buildResponsesRequestBody(alloc, "gpt-5.2", "[]", &messages, options);
    defer alloc.free(responses_body);
    var responses_parsed = try std.json.parseFromSlice(std.json.Value, alloc, responses_body, .{});
    defer responses_parsed.deinit();
    const responses_input = responses_parsed.value.object.get("input").?.array.items;
    try testing.expectEqual(@as(usize, 1), responses_input.len);
    const responses_parts = responses_input[0].object.get("content").?.array.items;
    try testing.expectEqual(@as(usize, 2), responses_parts.len);
    try testing.expectEqualStrings("inspect", responses_parts[0].object.get("text").?.string);
    try testing.expectEqualStrings(expected_url, responses_parts[1].object.get("image_url").?.string);

    const chat_body = try buildChatCompletionsRequestBody(alloc, "grok-4", "[]", &messages, options);
    defer alloc.free(chat_body);
    var chat_parsed = try std.json.parseFromSlice(std.json.Value, alloc, chat_body, .{});
    defer chat_parsed.deinit();
    const chat_parts = chat_parsed.value.object
        .get("messages").?.array.items[1].object
        .get("content").?.array.items;
    try testing.expectEqual(@as(usize, 2), chat_parts.len);
    try testing.expectEqualStrings(
        expected_url,
        chat_parts[1].object.get("image_url").?.object.get("url").?.string,
    );
}

test "verified snapshots require a final user message without inline images" {
    const alloc = testing.allocator;
    var bytes = png_fixture_bytes.*;
    const snapshots = [_]image_attachments.VerifiedSnapshot{.{
        .bytes = &bytes,
        .media_type = "image/png",
    }};
    const options = BuildOptions{ .verified_images = &snapshots };

    const trailing_assistant = [_]types.ChatMessage{
        .{ .role = .user, .content = "inspect" },
        .{ .role = .assistant, .content = "sure" },
    };
    try testing.expectError(
        error.InvalidRequestHistory,
        buildResponsesRequestBody(alloc, "gpt-5.2", "[]", &trailing_assistant, options),
    );
    try testing.expectError(
        error.InvalidRequestHistory,
        buildChatCompletionsRequestBody(alloc, "grok-4", "[]", &trailing_assistant, options),
    );
    try testing.expectError(
        error.InvalidRequestHistory,
        buildResponsesRequestBody(alloc, "gpt-5.2", "[]", &.{}, options),
    );
}

test "structured output reaches the matching field on each wire" {
    const alloc = testing.allocator;
    const messages = [_]types.ChatMessage{.{ .role = .user, .content = "inspect" }};
    const options = BuildOptions{
        .response_format = .{
            .name = "fx_vision_evidence",
            .description = "Evidence \"only\"",
            .schema_json = "{\"type\":\"object\",\"additionalProperties\":false}",
        },
    };

    const responses_body = try buildResponsesRequestBody(alloc, "gpt-5.2", "[]", &messages, options);
    defer alloc.free(responses_body);
    var responses_parsed = try std.json.parseFromSlice(std.json.Value, alloc, responses_body, .{});
    defer responses_parsed.deinit();
    const format = responses_parsed.value.object.get("text").?.object.get("format").?.object;
    try testing.expectEqualStrings("json_schema", format.get("type").?.string);
    try testing.expectEqualStrings("fx_vision_evidence", format.get("name").?.string);
    try testing.expectEqualStrings("Evidence \"only\"", format.get("description").?.string);
    try testing.expect(!format.get("strict").?.bool);
    try testing.expectEqualStrings("object", format.get("schema").?.object.get("type").?.string);
    try testing.expect(!format.get("schema").?.object.get("additionalProperties").?.bool);

    const chat_body = try buildChatCompletionsRequestBody(alloc, "grok-4", "[]", &messages, options);
    defer alloc.free(chat_body);
    var chat_parsed = try std.json.parseFromSlice(std.json.Value, alloc, chat_body, .{});
    defer chat_parsed.deinit();
    const response_format = chat_parsed.value.object.get("response_format").?.object;
    try testing.expectEqualStrings("json_schema", response_format.get("type").?.string);
    const json_schema = response_format.get("json_schema").?.object;
    try testing.expectEqualStrings("fx_vision_evidence", json_schema.get("name").?.string);
    try testing.expectEqualStrings("Evidence \"only\"", json_schema.get("description").?.string);
    try testing.expect(!json_schema.get("strict").?.bool);
    try testing.expectEqualStrings("object", json_schema.get("schema").?.object.get("type").?.string);

    const invalid = BuildOptions{
        .response_format = .{ .name = "bad", .schema_json = "not json" },
    };
    try testing.expectError(
        error.InvalidStructuredResponseSchema,
        buildResponsesRequestBody(alloc, "gpt-5.2", "[]", &messages, invalid),
    );
    try testing.expectError(
        error.InvalidStructuredResponseSchema,
        buildChatCompletionsRequestBody(alloc, "grok-4", "[]", &messages, invalid),
    );

    const plain = try buildResponsesRequestBody(alloc, "gpt-5.2", "[]", &messages, .{});
    defer alloc.free(plain);
    var plain_parsed = try std.json.parseFromSlice(std.json.Value, alloc, plain, .{});
    defer plain_parsed.deinit();
    try testing.expect(plain_parsed.value.object.get("text") == null);
}

test "a required tool name overrides the tool choice on both wires" {
    const alloc = testing.allocator;
    const messages = [_]types.ChatMessage{.{ .role = .user, .content = "look" }};
    const options = BuildOptions{ .tool_choice = .none, .required_tool_name = "vision" };

    const responses_body = try buildResponsesRequestBody(alloc, "gpt-5.2", "[]", &messages, options);
    defer alloc.free(responses_body);
    var responses_parsed = try std.json.parseFromSlice(std.json.Value, alloc, responses_body, .{});
    defer responses_parsed.deinit();
    const responses_choice = responses_parsed.value.object.get("tool_choice").?.object;
    try testing.expectEqualStrings("function", responses_choice.get("type").?.string);
    try testing.expectEqualStrings("vision", responses_choice.get("name").?.string);

    const chat_body = try buildChatCompletionsRequestBody(alloc, "grok-4", "[]", &messages, options);
    defer alloc.free(chat_body);
    var chat_parsed = try std.json.parseFromSlice(std.json.Value, alloc, chat_body, .{});
    defer chat_parsed.deinit();
    const chat_choice = chat_parsed.value.object.get("tool_choice").?.object;
    try testing.expectEqualStrings("function", chat_choice.get("type").?.string);
    try testing.expectEqualStrings(
        "vision",
        chat_choice.get("function").?.object.get("name").?.string,
    );

    const empty = BuildOptions{ .required_tool_name = "" };
    try testing.expectError(
        error.InvalidRequiredToolName,
        buildResponsesRequestBody(alloc, "gpt-5.2", "[]", &messages, empty),
    );
    try testing.expectError(
        error.InvalidRequiredToolName,
        buildChatCompletionsRequestBody(alloc, "grok-4", "[]", &messages, empty),
    );
}

const Capture = struct {
    content: std.ArrayList(u8) = .empty,
    reasoning: std.ArrayList(u8) = .empty,
    tool_input: std.ArrayList(u8) = .empty,
    starts: std.ArrayList(u8) = .empty,
    failed: bool = false,

    fn deinit(self: *Capture) void {
        self.content.deinit(testing.allocator);
        self.reasoning.deinit(testing.allocator);
        self.tool_input.deinit(testing.allocator);
        self.starts.deinit(testing.allocator);
    }

    fn onContent(raw: *anyopaque, chunk: []const u8) void {
        const self: *Capture = @ptrCast(@alignCast(raw));
        self.content.appendSlice(testing.allocator, chunk) catch {
            self.failed = true;
        };
    }

    fn onReasoning(raw: *anyopaque, chunk: []const u8) void {
        const self: *Capture = @ptrCast(@alignCast(raw));
        self.reasoning.appendSlice(testing.allocator, chunk) catch {
            self.failed = true;
        };
    }

    fn onToolInput(raw: *anyopaque, chunk: []const u8) void {
        const self: *Capture = @ptrCast(@alignCast(raw));
        self.tool_input.appendSlice(testing.allocator, chunk) catch {
            self.failed = true;
        };
    }

    fn onToolStart(raw: *anyopaque, tool_id: []const u8, tool_name: []const u8, _: ?[]const u8) void {
        const self: *Capture = @ptrCast(@alignCast(raw));
        self.starts.appendSlice(testing.allocator, tool_id) catch {
            self.failed = true;
        };
        self.starts.append(testing.allocator, ':') catch {
            self.failed = true;
        };
        self.starts.appendSlice(testing.allocator, tool_name) catch {
            self.failed = true;
        };
        self.starts.append(testing.allocator, ';') catch {
            self.failed = true;
        };
    }
};

fn runResponsesStream(
    payload: []const u8,
    capture: *Capture,
    cancel_flag: *std.atomic.Value(bool),
    content_capture_limit: ?usize,
) !types.GatewayCompletion {
    var reader = std.Io.Reader.fixed(payload);
    return consumeResponsesSseStream(
        testing.allocator,
        &reader,
        capture,
        Capture.onContent,
        Capture.onToolStart,
        Capture.onReasoning,
        Capture.onToolInput,
        cancel_flag,
        content_capture_limit,
    );
}

fn runChatStream(
    payload: []const u8,
    capture: *Capture,
    cancel_flag: *std.atomic.Value(bool),
    content_capture_limit: ?usize,
) !types.GatewayCompletion {
    var reader = std.Io.Reader.fixed(payload);
    return consumeChatCompletionsSseStream(
        testing.allocator,
        &reader,
        capture,
        Capture.onContent,
        Capture.onToolStart,
        Capture.onReasoning,
        Capture.onToolInput,
        cancel_flag,
        content_capture_limit,
    );
}

test "responses stream captures text, reasoning, usage, and generation id" {
    const payload =
        "event: response.created\n" ++
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_123\"}}\n" ++
        "\n" ++
        ": keep-alive\n" ++
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"thinking\"}\n" ++
        "\n" ++
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello \"}\n" ++
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"world\"}\n" ++
        "\n" ++
        "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":11,\"output_tokens\":4}}}\n" ++
        "\n" ++
        "data: [DONE]\n\n";

    var capture: Capture = .{};
    defer capture.deinit();
    var cancel_flag = std.atomic.Value(bool).init(false);

    var completion = try runResponsesStream(payload, &capture, &cancel_flag, null);
    defer freeCompletion(testing.allocator, &completion);

    try testing.expect(!capture.failed);
    try testing.expectEqualStrings("Hello world", completion.content.?);
    try testing.expectEqualStrings("Hello world", capture.content.items);
    try testing.expectEqualStrings("thinking", capture.reasoning.items);
    try testing.expectEqualStrings("resp_123", completion.generation_id.?);
    try testing.expectEqual(types.ProviderFinishReason.stop, completion.finish_reason.?);
    try testing.expectEqual(@as(u64, 11), completion.usage.input_tokens.?);
    try testing.expectEqual(@as(u64, 4), completion.usage.output_tokens.?);
    try testing.expectEqual(@as(usize, 0), completion.tool_calls.len);
}

test "responses stream assembles a function call from added, delta, and done" {
    const payload =
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_a\",\"name\":\"read_file\"}}\n\n" ++
        "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"output_index\":0,\"delta\":\"{\\\"path\\\":\"}\n\n" ++
        "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"output_index\":0,\"delta\":\"\\\"a.zig\\\"}\"}\n\n" ++
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_a\",\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"a.zig\\\"}\"}}\n\n" ++
        "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":2}}}\n\n" ++
        "data: [DONE]\n\n";

    var capture: Capture = .{};
    defer capture.deinit();
    var cancel_flag = std.atomic.Value(bool).init(false);

    var completion = try runResponsesStream(payload, &capture, &cancel_flag, null);
    defer freeCompletion(testing.allocator, &completion);

    try testing.expect(!capture.failed);
    try testing.expectEqualStrings("call_a:read_file;", capture.starts.items);
    try testing.expectEqualStrings("{\"path\":\"a.zig\"}", capture.tool_input.items);
    try testing.expectEqual(@as(usize, 1), completion.tool_calls.len);
    try testing.expectEqualStrings("call_a", completion.tool_calls[0].id);
    try testing.expectEqualStrings("read_file", completion.tool_calls[0].name);
    try testing.expectEqualStrings("{\"path\":\"a.zig\"}", completion.tool_calls[0].arguments_json);
    try testing.expectEqual(
        types.ToolArgumentIntegrity.valid,
        completion.tool_calls[0].argument_integrity,
    );
    try testing.expectEqual(types.ProviderFinishReason.tool_calls, completion.finish_reason.?);
}

test "responses stream reports failed and incomplete terminal states" {
    var capture: Capture = .{};
    defer capture.deinit();
    var cancel_flag = std.atomic.Value(bool).init(false);

    const failed_payload =
        "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"code\":\"server_error\",\"message\":\"upstream exploded\"}}}\n\n" ++
        "data: [DONE]\n\n";
    var failed = try runResponsesStream(failed_payload, &capture, &cancel_flag, null);
    defer freeCompletion(testing.allocator, &failed);
    try testing.expectEqual(types.ProviderFinishReason.provider_error, failed.finish_reason.?);
    try testing.expectEqualStrings("upstream exploded", failed.provider_failure_detail.?);

    const incomplete_payload =
        "data: {\"type\":\"response.incomplete\",\"response\":{\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n" ++
        "data: [DONE]\n\n";
    var incomplete = try runResponsesStream(incomplete_payload, &capture, &cancel_flag, null);
    defer freeCompletion(testing.allocator, &incomplete);
    try testing.expectEqual(types.ProviderFinishReason.length, incomplete.finish_reason.?);

    const error_payload =
        "data: {\"type\":\"error\",\"message\":\"stream aborted\"}\n\n";
    var errored = try runResponsesStream(error_payload, &capture, &cancel_flag, null);
    defer freeCompletion(testing.allocator, &errored);
    try testing.expectEqualStrings("stream aborted", errored.provider_failure_detail.?);
    try testing.expect(errored.finish_reason == null);
}

test "responses stream skips malformed and unknown events" {
    const payload =
        "data: not json at all\n\n" ++
        "data: [1,2,3]\n\n" ++
        "data: {\"type\":\"response.future_event\",\"delta\":\"ignored\"}\n\n" ++
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n" ++
        "data: [DONE]\n\n";

    var capture: Capture = .{};
    defer capture.deinit();
    var cancel_flag = std.atomic.Value(bool).init(false);

    var completion = try runResponsesStream(payload, &capture, &cancel_flag, null);
    defer freeCompletion(testing.allocator, &completion);

    try testing.expectEqualStrings("ok", completion.content.?);
    try testing.expectEqualStrings("ok", capture.content.items);
}

test "responses stream truncates captured content but not callbacks" {
    const payload =
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"12345\"}\n\n" ++
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"67890\"}\n\n" ++
        "data: [DONE]\n\n";

    var capture: Capture = .{};
    defer capture.deinit();
    var cancel_flag = std.atomic.Value(bool).init(false);

    var completion = try runResponsesStream(payload, &capture, &cancel_flag, 7);
    defer freeCompletion(testing.allocator, &completion);

    try testing.expectEqualStrings("1234567", completion.content.?);
    try testing.expectEqualStrings("1234567890", capture.content.items);
}

test "responses stream returns Cancelled when the flag is already set" {
    const payload = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"x\"}\n\n";
    var capture: Capture = .{};
    defer capture.deinit();
    var cancel_flag = std.atomic.Value(bool).init(true);

    try testing.expectError(
        error.Cancelled,
        runResponsesStream(payload, &capture, &cancel_flag, null),
    );
}

test "responses stream ends at end of input without a done marker" {
    const payload = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"tail\"}\n\n";
    var capture: Capture = .{};
    defer capture.deinit();
    var cancel_flag = std.atomic.Value(bool).init(false);

    var completion = try runResponsesStream(payload, &capture, &cancel_flag, null);
    defer freeCompletion(testing.allocator, &completion);
    try testing.expectEqualStrings("tail", completion.content.?);
    try testing.expect(completion.finish_reason == null);
}

test "chat completions stream captures text, reasoning, usage, finish reason, and id" {
    const payload =
        "data: {\"id\":\"chatcmpl_9\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n" ++
        "data: {\"id\":\"chatcmpl_9\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"hmm\"}}]}\n\n" ++
        "data: {\"id\":\"chatcmpl_9\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi \"}}]}\n\n" ++
        "data: {\"id\":\"chatcmpl_9\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"there\"},\"finish_reason\":\"stop\"}]}\n\n" ++
        "data: {\"id\":\"chatcmpl_9\",\"choices\":[],\"usage\":{\"prompt_tokens\":21,\"completion_tokens\":3}}\n\n" ++
        "data: [DONE]\n\n";

    var capture: Capture = .{};
    defer capture.deinit();
    var cancel_flag = std.atomic.Value(bool).init(false);

    var completion = try runChatStream(payload, &capture, &cancel_flag, null);
    defer freeCompletion(testing.allocator, &completion);

    try testing.expect(!capture.failed);
    try testing.expectEqualStrings("Hi there", completion.content.?);
    try testing.expectEqualStrings("Hi there", capture.content.items);
    try testing.expectEqualStrings("hmm", capture.reasoning.items);
    try testing.expectEqualStrings("chatcmpl_9", completion.generation_id.?);
    try testing.expectEqual(types.ProviderFinishReason.stop, completion.finish_reason.?);
    try testing.expectEqual(@as(u64, 21), completion.usage.input_tokens.?);
    try testing.expectEqual(@as(u64, 3), completion.usage.output_tokens.?);
}

test "chat completions stream assembles indexed tool calls" {
    const payload =
        "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_x\",\"type\":\"function\",\"function\":{\"name\":\"run_command\",\"arguments\":\"\"}}]}}]}\n\n" ++
        "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"cmd\\\":\"}}]}}]}\n\n" ++
        "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"ls\\\"}\"}}]}}]}\n\n" ++
        "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_y\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{}\"}}]}}]}\n\n" ++
        "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n" ++
        "data: [DONE]\n\n";

    var capture: Capture = .{};
    defer capture.deinit();
    var cancel_flag = std.atomic.Value(bool).init(false);

    var completion = try runChatStream(payload, &capture, &cancel_flag, null);
    defer freeCompletion(testing.allocator, &completion);

    try testing.expect(!capture.failed);
    try testing.expectEqualStrings("call_x:run_command;call_y:read_file;", capture.starts.items);
    try testing.expectEqualStrings("{\"cmd\":\"ls\"}{}", capture.tool_input.items);
    try testing.expectEqual(types.ProviderFinishReason.tool_calls, completion.finish_reason.?);
    try testing.expectEqual(@as(usize, 2), completion.tool_calls.len);
    try testing.expectEqualStrings("call_x", completion.tool_calls[0].id);
    try testing.expectEqualStrings("run_command", completion.tool_calls[0].name);
    try testing.expectEqualStrings("{\"cmd\":\"ls\"}", completion.tool_calls[0].arguments_json);
    try testing.expectEqualStrings("call_y", completion.tool_calls[1].id);
    try testing.expectEqualStrings("{}", completion.tool_calls[1].arguments_json);
}

test "chat completions stream maps every finish reason and defaults empty arguments" {
    const cases = [_]struct {
        raw: []const u8,
        expected: types.ProviderFinishReason,
    }{
        .{ .raw = "stop", .expected = .stop },
        .{ .raw = "length", .expected = .length },
        .{ .raw = "tool_calls", .expected = .tool_calls },
        .{ .raw = "content_filter", .expected = .content_filter },
        .{ .raw = "function_call", .expected = .other },
    };

    for (cases) |case| {
        var buffer: [256]u8 = undefined;
        const payload = try std.fmt.bufPrint(
            &buffer,
            "data: {{\"id\":\"c\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{s}\"}}]}}\n\ndata: [DONE]\n\n",
            .{case.raw},
        );

        var capture: Capture = .{};
        defer capture.deinit();
        var cancel_flag = std.atomic.Value(bool).init(false);

        var completion = try runChatStream(payload, &capture, &cancel_flag, null);
        defer freeCompletion(testing.allocator, &completion);
        try testing.expectEqual(case.expected, completion.finish_reason.?);
    }

    const empty_arguments =
        "data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_z\",\"function\":{\"name\":\"noop\"}}]}}]}\n\n" ++
        "data: [DONE]\n\n";
    var capture: Capture = .{};
    defer capture.deinit();
    var cancel_flag = std.atomic.Value(bool).init(false);
    var completion = try runChatStream(empty_arguments, &capture, &cancel_flag, null);
    defer freeCompletion(testing.allocator, &completion);
    try testing.expectEqualStrings("{}", completion.tool_calls[0].arguments_json);
}

test "chat completions stream skips malformed data and truncates captured content" {
    const payload =
        "data: {oops\n\n" ++
        "data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"abc\"}}]}\n\n" ++
        "data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"def\"}}]}\n\n" ++
        "data: [DONE]\n\n";

    var capture: Capture = .{};
    defer capture.deinit();
    var cancel_flag = std.atomic.Value(bool).init(false);

    var completion = try runChatStream(payload, &capture, &cancel_flag, 4);
    defer freeCompletion(testing.allocator, &completion);

    try testing.expectEqualStrings("abcd", completion.content.?);
    try testing.expectEqualStrings("abcdef", capture.content.items);
}

test "chat completions stream returns Cancelled when the flag is already set" {
    const payload = "data: {\"id\":\"c\",\"choices\":[]}\n\n";
    var capture: Capture = .{};
    defer capture.deinit();
    var cancel_flag = std.atomic.Value(bool).init(true);

    try testing.expectError(
        error.Cancelled,
        runChatStream(payload, &capture, &cancel_flag, null),
    );
}
