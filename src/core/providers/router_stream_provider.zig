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
const provider_credentials = @import("provider_credentials.zig");
const registry = @import("registry.zig");

const Allocator = std.mem.Allocator;

const direct_payload_prefix = "{\"model\":";

/// True when a build request can be served on a direct wire protocol.
/// Vision-required turns, structured output, and dynamic tool schemas stay
/// on the gateway. Optional vision routes directly without the vision tool;
/// turns that embed image attachments fall back through the codec's
/// UnsupportedImageAttachments rejection.
fn buildEligible(request: agent_stream_provider.BuildRequest) bool {
    return request.vision_mode != .required and
        request.verified_images == null and
        request.response_format == null and
        request.selected_dynamic_tool_schemas.len == 0;
}

fn routeForModel(model: []const u8) ?struct {
    def: *const registry.Def,
    kind: provider_credentials.Kind,
} {
    const def = registry.byModel(model) orelse return null;
    const kind = provider_credentials.preferredKind(def) orelse return null;
    return .{ .def = def, .kind = kind };
}

fn buildDirect(
    alloc: Allocator,
    def: *const registry.Def,
    kind: provider_credentials.Kind,
    request: agent_stream_provider.BuildRequest,
) anyerror!?[]u8 {
    const wire = switch (kind) {
        .api_key => def.api_wire,
        .oauth => def.oauth_wire,
    };
    const bare_model = registry.bareModel(def, request.model);
    const body = switch (wire) {
        .openai_responses => openai_json.buildResponsesRequestBody(
            alloc,
            bare_model,
            request.serialized_tools,
            request.messages,
            request.provider_options,
            request.tool_choice,
            request.max_output_tokens,
        ),
        .openai_chat_completions => openai_json.buildChatCompletionsRequestBody(
            alloc,
            bare_model,
            request.serialized_tools,
            request.messages,
            request.provider_options,
            request.tool_choice,
            request.max_output_tokens,
        ),
    } catch |err| switch (err) {
        error.UnsupportedImageAttachments => return null,
        else => return err,
    };
    return body;
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
            if (buildEligible(request)) {
                if (routeForModel(request.model)) |route| {
                    if (try buildDirect(alloc, route.def, route.kind, request)) |body| {
                        return body;
                    }
                }
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

test "build eligibility excludes vision and structured output requests" {
    const base = agent_stream_provider.BuildRequest{
        .model = "openai/gpt-5.2",
        .serialized_tools = "[]",
        .messages = &.{},
        .tool_choice = .auto,
        .provider_options = .{},
    };
    try std.testing.expect(buildEligible(base));

    var vision = base;
    vision.vision_mode = .optional;
    try std.testing.expect(!buildEligible(vision));

    var structured = base;
    structured.response_format = .{ .name = "n", .description = "d", .schema_json = "{}" };
    try std.testing.expect(!buildEligible(structured));

    var dynamic = base;
    dynamic.selected_dynamic_tool_schemas = &.{"{}"};
    try std.testing.expect(!buildEligible(dynamic));
}

test "models without provider credentials fall through" {
    try std.testing.expect(routeForModel("anthropic/claude-opus-4.8") == null);
    // openai/xai models only route when a credential exists; unit test
    // environments have neither env keys nor session files for them unless
    // the developer machine does, so only assert the unknown-prefix case.
}
