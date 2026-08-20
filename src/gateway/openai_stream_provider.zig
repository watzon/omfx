//! omfx fork-owned HTTP transport for direct OpenAI-compatible providers.
//!
//! Streams a prepared request body to a provider chat endpoint and hands the
//! SSE response body to a wire-specific consumer (see openai_json.zig). The
//! Vercel gateway transport in client.zig stays untouched; this is the
//! parallel path the provider router uses for direct credentials.

const std = @import("std");
const agent_stream_provider = @import("../core/agent/stream_provider.zig");
const debug_trace = @import("../core/shared/debug_trace.zig");
const io_mod = @import("../core/shared/io.zig");
const types = @import("../core/shared/types.zig");
const gateway_client = @import("client.zig");

const Allocator = std.mem.Allocator;

const transfer_buffer_bytes: usize = 64 * 1024;
const cancel_poll_interval_ns: u64 = 20 * std.time.ns_per_ms;
const retry_base_delay_ns: u64 = 150 * std.time.ns_per_ms;

/// Consumes one SSE response body into a completion. Implementations live in
/// openai_json.zig; the signature mirrors client.zig's gateway consumer.
pub const ConsumeFn = *const fn (
    alloc: Allocator,
    reader: *std.Io.Reader,
    callback_ctx: *anyopaque,
    on_content_chunk: agent_stream_provider.StreamCallback,
    on_tool_start: ?agent_stream_provider.ToolStartCallback,
    on_reasoning_chunk: ?agent_stream_provider.StreamCallback,
    on_tool_input_chunk: ?agent_stream_provider.StreamCallback,
    cancel_flag: *std.atomic.Value(bool),
    content_capture_limit: ?usize,
) anyerror!types.GatewayCompletion;

pub const RequestSpec = struct {
    url: []const u8,
    bearer_token: []const u8,
    /// ChatGPT-Account-ID header for the OpenAI subscription backend.
    account_id: ?[]const u8 = null,
    /// The ChatGPT Codex backend requires client identification headers.
    chatgpt_backend: bool = false,
};

pub fn stream(
    alloc: Allocator,
    spec: RequestSpec,
    request: agent_stream_provider.Request,
    consume: ConsumeFn,
) anyerror!agent_stream_provider.Result {
    if (request.cancel_flag.load(.seq_cst)) return error.Cancelled;
    if (!urlAllowed(spec.url)) return error.UntrustedProviderUrl;

    const uri = try std.Uri.parse(spec.url);
    const auth_header = try std.fmt.allocPrint(alloc, "Bearer {s}", .{spec.bearer_token});
    defer alloc.free(auth_header);

    var extra_headers_buf: [4]std.http.Header = undefined;
    var extra_len: usize = 0;
    if (spec.account_id) |account_id| {
        if (account_id.len > 0) {
            extra_headers_buf[extra_len] = .{ .name = "ChatGPT-Account-ID", .value = account_id };
            extra_len += 1;
        }
    }
    if (spec.chatgpt_backend) {
        extra_headers_buf[extra_len] = .{ .name = "OpenAI-Beta", .value = "responses=experimental" };
        extra_len += 1;
        extra_headers_buf[extra_len] = .{ .name = "originator", .value = "codex_cli_rs" };
        extra_len += 1;
        if (request.session_id) |session_id| {
            if (session_id.len > 0) {
                extra_headers_buf[extra_len] = .{ .name = "session_id", .value = session_id };
                extra_len += 1;
            }
        }
    }
    const extra_headers = extra_headers_buf[0..extra_len];

    const attempt_limit = switch (request.provider_attempt_owner) {
        .agent => 1,
        .transport => @max(request.retry_count, 1),
    };

    var attempt: usize = 0;
    var delivery_ambiguous = false;
    while (attempt < attempt_limit) : (attempt += 1) {
        if (request.cancel_flag.load(.seq_cst)) return error.Cancelled;
        var client: std.http.Client = .{ .allocator = alloc, .io = io_mod.getIo() };
        defer client.deinit();

        debug_trace.eventf("provider", "before_request_open", request.trace_ctx, "attempt={d} payload_bytes={d}", .{ attempt + 1, request.payload.len });
        var req = client.request(.POST, uri, .{
            .headers = .{
                .content_type = .{ .override = "application/json" },
                .authorization = .{ .override = auth_header },
                .accept_encoding = .omit,
                .user_agent = .{ .override = gateway_client.user_agent },
            },
            .extra_headers = extra_headers,
            .keep_alive = false,
            .redirect_behavior = .unhandled,
        }) catch |err| {
            debug_trace.eventf("provider", "request_open_error", request.trace_ctx, "attempt={d} err={s}", .{ attempt + 1, @errorName(err) });
            if (request.cancel_flag.load(.seq_cst)) return error.Cancelled;
            if (attempt + 1 < attempt_limit and gateway_client.isRetryableGatewayError(err)) {
                try sleepRetry((attempt + 1) * retry_base_delay_ns, request.cancel_flag);
                continue;
            }
            return err;
        };
        defer req.deinit();

        var cancel_watch_done = std.atomic.Value(bool).init(false);
        const cancel_watcher: ?std.Thread = if (req.connection) |conn|
            std.Thread.spawn(.{}, watchCancel, .{
                &cancel_watch_done,
                request.cancel_flag,
                conn.stream_writer.stream,
            }) catch null
        else
            null;
        defer {
            cancel_watch_done.store(true, .seq_cst);
            if (cancel_watcher) |thread| thread.join();
        }
        if (request.cancel_flag.load(.seq_cst)) return error.Cancelled;

        req.transfer_encoding = .{ .content_length = request.payload.len };
        request.delivery.markPossiblySent();
        var send_buf: [8192]u8 = undefined;
        sendPayload(&req, &send_buf, request.payload) catch |err| {
            return connectedFailure(request.cancel_flag, err);
        };

        var response = req.receiveHead(&.{}) catch |err| {
            const mapped = connectedFailure(request.cancel_flag, err);
            if (mapped != error.Cancelled and
                request.provider_attempt_owner == .transport and
                gateway_client.isRetryableGatewayError(mapped) and
                attempt + 1 < attempt_limit)
            {
                delivery_ambiguous = true;
                try sleepRetry((attempt + 1) * retry_base_delay_ns, request.cancel_flag);
                continue;
            }
            return mapped;
        };
        debug_trace.eventf("provider", "after_receive_head", request.trace_ctx, "attempt={d} status={d}", .{ attempt + 1, @intFromEnum(response.head.status) });

        if (response.head.status != .ok) {
            const status = response.head.status;
            if (@intFromEnum(status) >= 500) delivery_ambiguous = true;
            var err_out: std.Io.Writer.Allocating = .init(alloc);
            defer err_out.deinit();
            var err_buf: [4096]u8 = undefined;
            const err_reader = response.reader(&err_buf);
            _ = err_reader.streamRemaining(&err_out.writer) catch {};
            if (request.cancel_flag.load(.seq_cst)) return error.Cancelled;
            if (request.provider_attempt_owner == .transport and
                retryableStatus(status) and
                attempt + 1 < attempt_limit)
            {
                try sleepRetry((attempt + 1) * retry_base_delay_ns, request.cancel_flag);
                continue;
            }
            return .{
                .status = status,
                .err_body = err_out.toOwnedSlice() catch null,
                .completion = .{ .delivery_ambiguous = delivery_ambiguous },
                .ownership = .owned,
            };
        }

        var transfer_buf: [transfer_buffer_bytes]u8 = undefined;
        const body_reader = response.reader(&transfer_buf);
        var completion = consume(
            alloc,
            body_reader,
            request.callback_ctx,
            request.on_content_chunk,
            request.on_tool_start,
            request.on_reasoning_chunk,
            request.on_tool_input_chunk,
            request.cancel_flag,
            request.content_capture_limit,
        ) catch |err| {
            return connectedFailure(request.cancel_flag, err);
        };
        if (request.cancel_flag.load(.seq_cst)) {
            freeCompletion(alloc, &completion);
            return error.Cancelled;
        }
        completion.delivery_ambiguous = delivery_ambiguous;
        return .{
            .status = .ok,
            .completion = completion,
            .ownership = .owned,
        };
    }
    return error.HttpConnectionClosing;
}

fn sendPayload(req: *std.http.Client.Request, send_buf: []u8, payload: []const u8) !void {
    var body_writer = try req.sendBodyUnflushed(send_buf);
    try body_writer.writer.writeAll(payload);
    try body_writer.end();
    try req.connection.?.flush();
}

fn connectedFailure(cancel_flag: *std.atomic.Value(bool), err: anyerror) anyerror {
    if (cancel_flag.load(.seq_cst)) return error.Cancelled;
    return err;
}

fn watchCancel(
    done: *std.atomic.Value(bool),
    cancel_flag: *std.atomic.Value(bool),
    socket: std.Io.net.Stream,
) void {
    while (!done.load(.seq_cst)) {
        if (cancel_flag.load(.seq_cst)) {
            socket.shutdown(io_mod.getIo(), .both) catch {};
            return;
        }
        io_mod.sleep(cancel_poll_interval_ns);
    }
}

fn sleepRetry(delay_ns: u64, cancel_flag: *std.atomic.Value(bool)) !void {
    const deadline = std.Io.Clock.Timestamp.fromNow(io_mod.getIo(), .{
        .clock = .awake,
        .raw = .fromNanoseconds(@intCast(delay_ns)),
    });
    while (true) {
        if (cancel_flag.load(.seq_cst)) return error.Cancelled;
        const now = std.Io.Clock.Timestamp.now(io_mod.getIo(), .awake);
        if (!std.Io.Clock.Timestamp.compare(now, .lt, deadline)) return;
        const remaining_ns = now.raw.durationTo(deadline.raw).toNanoseconds();
        const step_ns: i96 = @min(@as(i96, cancel_poll_interval_ns), remaining_ns);
        if (step_ns <= 0) return;
        io_mod.sleep(@intCast(step_ns));
    }
}

fn retryableStatus(status: std.http.Status) bool {
    return switch (@intFromEnum(status)) {
        408, 429, 500, 502, 503, 504 => true,
        else => false,
    };
}

/// Bearer tokens leave the process only over HTTPS, or over loopback HTTP
/// for test fixtures wired through the base-url env overrides.
fn urlAllowed(url: []const u8) bool {
    const uri = std.Uri.parse(url) catch return false;
    if (std.ascii.eqlIgnoreCase(uri.scheme, "https")) return true;
    return gateway_client.isLoopbackHttpUrl(url);
}

fn freeCompletion(alloc: Allocator, completion: *types.GatewayCompletion) void {
    if (completion.content) |content| alloc.free(@constCast(content));
    if (completion.generation_id) |id| alloc.free(@constCast(id));
    if (completion.provider_failure_detail) |detail| alloc.free(@constCast(detail));
    types.freeToolCallSlice(alloc, @constCast(completion.tool_calls));
    completion.* = .{};
}

test "provider urls require https or loopback http" {
    try std.testing.expect(urlAllowed("https://api.openai.com/v1/responses"));
    try std.testing.expect(urlAllowed("http://127.0.0.1:43123/responses"));
    try std.testing.expect(!urlAllowed("http://api.openai.com/v1/responses"));
    try std.testing.expect(!urlAllowed("not a url"));
}

test "retryable statuses cover throttling and transient server failures" {
    try std.testing.expect(retryableStatus(@enumFromInt(429)));
    try std.testing.expect(retryableStatus(@enumFromInt(503)));
    try std.testing.expect(!retryableStatus(@enumFromInt(401)));
    try std.testing.expect(!retryableStatus(@enumFromInt(404)));
}
