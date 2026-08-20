//! omfx fork-owned provider router.
//!
//! Wraps the upstream gateway stream provider: a request whose model id
//! belongs to a direct provider with a live credential (see
//! provider_credentials.zig) is built and streamed on that provider's own
//! wire protocol; everything else falls through to the untouched gateway
//! path. build and stream are separate calls, so the routing decision is
//! re-derived at stream time from the payload shape: direct payloads start
//! with `{"model":`, gateway payloads with `{"prompt":`.

const std = @import("std");
const openai_json = @import("../../gateway/openai_json.zig");
const openai_stream = @import("../../gateway/openai_stream_provider.zig");
const agent_stream_provider = @import("../agent/stream_provider.zig");
const oauth_transport = @import("../auth/oauth_transport.zig");
const debug_trace = @import("../shared/debug_trace.zig");
const gateway_schema = @import("../tooling/gateway_schema.zig");
const tool_dispatch = @import("../tooling/tool_dispatch.zig");
const provider_credentials = @import("provider_credentials.zig");
const registry = @import("registry.zig");

const Allocator = std.mem.Allocator;

const direct_payload_prefix = "{\"model\":";

/// Mirrors the constraints buildAgentRequest enforces before building.
fn validateBuildShape(request: agent_stream_provider.BuildRequest) !void {
    if (request.verified_images != null) {
        if (request.response_format == null) return error.MissingStructuredResponseFormat;
        return;
    }
    if (request.response_format != null) return error.StructuredResponseRequiresVerifiedImages;
}

fn routeForModel(model: []const u8) ?struct {
    def: *const registry.Def,
    kind: provider_credentials.Kind,
} {
    const def = registry.byModel(model) orelse return null;
    const kind = provider_credentials.preferredKind(def) orelse return null;
    return .{ .def = def, .kind = kind };
}

fn writeVisionSchema(alloc: Allocator, tool_registry: tool_dispatch.Registry) ![]u8 {
    const vision_tool = tool_registry.lookup("vision") orelse return error.VisionToolNotRegistered;
    var out: std.Io.Writer.Allocating = .init(alloc);
    errdefer out.deinit();
    try gateway_schema.writeBuiltinFunctionSchema(alloc, &out.writer, vision_tool.gateway_schema);
    return out.toOwnedSlice();
}

/// Appends extra tool schema objects to a serialized tools JSON array.
fn mergeToolsJson(alloc: Allocator, serialized_tools: []const u8, extras: []const []const u8) ![]u8 {
    const trimmed = std.mem.trim(u8, serialized_tools, " \t\r\n");
    if (trimmed.len < 2 or trimmed[0] != '[' or trimmed[trimmed.len - 1] != ']') {
        return error.InvalidToolSchema;
    }
    const inner = std.mem.trim(u8, trimmed[1 .. trimmed.len - 1], " \t\r\n");
    var out: std.Io.Writer.Allocating = .init(alloc);
    errdefer out.deinit();
    try out.writer.writeByte('[');
    try out.writer.writeAll(inner);
    for (extras) |extra| {
        if (out.writer.buffered().len > 1) try out.writer.writeByte(',');
        try out.writer.writeAll(extra);
    }
    try out.writer.writeByte(']');
    return out.toOwnedSlice();
}

fn buildDirect(
    alloc: Allocator,
    def: *const registry.Def,
    kind: provider_credentials.Kind,
    request: agent_stream_provider.BuildRequest,
) anyerror![]u8 {
    if (request.budget) |budget| {
        if (budget.cancel_flag) |flag| {
            if (flag.load(.seq_cst)) return error.Cancelled;
        }
    }
    try validateBuildShape(request);

    const wire = switch (kind) {
        .api_key => def.api_wire,
        .oauth => def.oauth_wire,
    };
    const bare_model = registry.bareModel(def, request.model);

    var options: openai_json.BuildOptions = .{
        .provider_options = request.provider_options,
        .tool_choice = request.tool_choice,
        .max_output_tokens = request.max_output_tokens,
        .verified_images = request.verified_images,
    };
    if (request.response_format) |format| {
        options.response_format = .{
            .name = format.name,
            .description = format.description,
            .schema_json = format.schema_json,
        };
    }

    const vision_schema: ?[]u8 = if (request.vision_mode != .unavailable)
        try writeVisionSchema(alloc, request.tool_registry)
    else
        null;
    defer if (vision_schema) |schema| alloc.free(schema);

    var owned_tools: ?[]u8 = null;
    defer if (owned_tools) |tools| alloc.free(tools);
    var tools_json: []const u8 = request.serialized_tools;

    if (request.vision_mode == .required) {
        owned_tools = try std.fmt.allocPrint(alloc, "[{s}]", .{vision_schema.?});
        tools_json = owned_tools.?;
        options.required_tool_name = "vision";
    } else {
        var extras: std.ArrayList([]const u8) = .empty;
        defer extras.deinit(alloc);
        for (request.selected_dynamic_tool_schemas) |schema| try extras.append(alloc, schema);
        if (vision_schema) |schema| try extras.append(alloc, schema);
        if (extras.items.len > 0) {
            owned_tools = try mergeToolsJson(alloc, request.serialized_tools, extras.items);
            tools_json = owned_tools.?;
        }
    }

    return switch (wire) {
        .openai_responses => openai_json.buildResponsesRequestBody(
            alloc,
            bare_model,
            tools_json,
            request.messages,
            options,
        ),
        .openai_chat_completions => openai_json.buildChatCompletionsRequestBody(
            alloc,
            bare_model,
            tools_json,
            request.messages,
            options,
        ),
    };
}

fn streamDirect(
    alloc: Allocator,
    transport: oauth_transport.Provider,
    def: *const registry.Def,
    request: agent_stream_provider.Request,
) anyerror!agent_stream_provider.Result {
    var credential = provider_credentials.resolve(alloc, transport, def) catch |err| {
        debug_trace.logf("provider", "direct credential resolve failed provider={s} err={s}", .{ def.key, @errorName(err) });
        return err;
    };
    defer credential.deinit(alloc);

    const oauth = credential.kind == .oauth;
    const wire = if (oauth) def.oauth_wire else def.api_wire;
    const base = registry.baseUrl(def, oauth);
    const path = if (oauth) def.oauth_chat_path else def.api_chat_path;
    const url = try std.fmt.allocPrint(alloc, "{s}{s}", .{ base, path });
    defer alloc.free(url);

    const chatgpt_backend = oauth and std.mem.eql(u8, def.key, "openai");
    debug_trace.logf("provider", "direct stream provider={s} kind={t} wire={t}", .{ def.key, credential.kind, wire });
    return openai_stream.stream(alloc, .{
        .url = url,
        .bearer_token = credential.token,
        .account_id = credential.account_id,
        .chatgpt_backend = chatgpt_backend,
    }, request, switch (wire) {
        .openai_responses => openai_json.consumeResponsesSseStream,
        .openai_chat_completions => openai_json.consumeChatCompletionsSseStream,
    });
}

/// Wraps the gateway build fn: direct-provider models with credentials get a
/// provider-native payload; everything else falls through unchanged.
pub fn makeBuild(comptime fallback: agent_stream_provider.BuildFn) agent_stream_provider.BuildFn {
    const Wrapper = struct {
        fn build(
            context: ?*anyopaque,
            alloc: Allocator,
            request: agent_stream_provider.BuildRequest,
        ) anyerror![]u8 {
            if (routeForModel(request.model)) |route| {
                return buildDirect(alloc, route.def, route.kind, request);
            }
            return fallback(context, alloc, request);
        }
    };
    return Wrapper.build;
}

/// Wraps the gateway stream fn. A request routes directly only when the
/// model belongs to a credentialed provider AND the payload was built for a
/// direct wire, keeping build and stream decisions consistent.
pub fn makeStream(
    comptime fallback: agent_stream_provider.StreamFn,
    comptime transport: oauth_transport.Provider,
) agent_stream_provider.StreamFn {
    const Wrapper = struct {
        fn stream(
            context: ?*anyopaque,
            alloc: Allocator,
            request: agent_stream_provider.Request,
        ) anyerror!agent_stream_provider.Result {
            if (std.mem.startsWith(u8, request.payload, direct_payload_prefix)) {
                if (routeForModel(request.model)) |route| {
                    return streamDirect(alloc, transport, route.def, request);
                }
            }
            return fallback(context, alloc, request);
        }
    };
    return Wrapper.stream;
}

test "gateway payloads never route to a direct provider" {
    try std.testing.expect(!std.mem.startsWith(
        u8,
        "{\"prompt\":[],\"tools\":[]}",
        direct_payload_prefix,
    ));
    try std.testing.expect(std.mem.startsWith(
        u8,
        "{\"model\":\"grok-4\",\"stream\":true}",
        direct_payload_prefix,
    ));
}

test "build shape validation mirrors the gateway builder rules" {
    const base = agent_stream_provider.BuildRequest{
        .model = "openai/gpt-5.2",
        .serialized_tools = "[]",
        .messages = &.{},
        .tool_choice = .auto,
        .provider_options = .{},
    };
    try validateBuildShape(base);

    var structured = base;
    structured.response_format = .{ .name = "n", .description = "d", .schema_json = "{}" };
    try std.testing.expectError(
        error.StructuredResponseRequiresVerifiedImages,
        validateBuildShape(structured),
    );

    var images_only = base;
    images_only.verified_images = &.{};
    try std.testing.expectError(
        error.MissingStructuredResponseFormat,
        validateBuildShape(images_only),
    );
}

test "merging tool schemas keeps existing entries and appends extras" {
    const alloc = std.testing.allocator;
    const merged = try mergeToolsJson(alloc, "[{\"name\":\"a\"}]", &.{ "{\"name\":\"b\"}", "{\"name\":\"c\"}" });
    defer alloc.free(merged);
    try std.testing.expectEqualStrings("[{\"name\":\"a\"},{\"name\":\"b\"},{\"name\":\"c\"}]", merged);

    const from_empty = try mergeToolsJson(alloc, "[]", &.{"{\"name\":\"b\"}"});
    defer alloc.free(from_empty);
    try std.testing.expectEqualStrings("[{\"name\":\"b\"}]", from_empty);

    try std.testing.expectError(error.InvalidToolSchema, mergeToolsJson(alloc, "{}", &.{"{}"}));
}

test "models without provider credentials fall through" {
    try std.testing.expect(routeForModel("anthropic/claude-opus-4.8") == null);
    // openai/xai models only route when a credential exists; unit test
    // environments have neither env keys nor session files for them unless
    // the developer machine does, so only assert the unknown-prefix case.
}
