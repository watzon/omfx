//! omfx fork-owned interactive sign-in driver for direct providers.
//!
//! The onboarding and /setup pickers dispatch here. `start` runs the fast
//! phase of a provider OAuth flow (compose the URL, bind the callback
//! listener) and hands back the text the caller should display; a detached
//! worker thread then waits for the user to finish in the browser.
//! `takeOutcome` is polled once per frame and reports the terminal result.
//!
//! This module never touches the app struct: the caller owns all rendering,
//! browser opening, and notice writing.
//!
//! One sign-in runs at a time; the transient state is module-global because
//! the upstream app struct is not fork-owned. The worker publishes its
//! outcome with release ordering and `takeOutcome` consumes it with acquire.

const std = @import("std");
const oauth_transport = @import("../auth/oauth_transport.zig");
const debug_trace = @import("../shared/debug_trace.zig");
const oauth_flows = @import("oauth_flows.zig");
const registry = @import("registry.zig");

const Allocator = std.mem.Allocator;

const Phase = enum(u8) {
    idle,
    waiting,
    succeeded,
    failed,
};

const max_failure_name_bytes = 64;

var phase = std.atomic.Value(u8).init(@intFromEnum(Phase.idle));
var active_def: ?*const registry.Def = null;
var failure_name_buf: [max_failure_name_bytes]u8 = undefined;
var failure_name_len: usize = 0;

const Job = struct {
    transport: oauth_transport.Provider,
    def: *const registry.Def,
    pending: union(enum) {
        device: oauth_flows.DevicePending,
        pkce: oauth_flows.PkcePending,
    },
};

/// What the caller should show and open when a sign-in starts. `notice` is
/// owned by the allocator passed to `start`; `browser_url` borrows from it.
pub const Started = struct {
    notice: []u8,
    browser_url: []const u8,
};

pub const Outcome = union(enum) {
    /// Owned by the allocator passed to `takeOutcome`.
    succeeded: []u8,
    failed: []u8,
};

fn loadPhase() Phase {
    return @enumFromInt(phase.load(.acquire));
}

fn storePhase(value: Phase) void {
    phase.store(@intFromEnum(value), .release);
}

pub fn active() bool {
    return loadPhase() != .idle;
}

/// Runs the fast phase and spawns the waiter. Returns null when a sign-in is
/// already running. Errors from the provider flow propagate to the caller.
pub fn start(
    alloc: Allocator,
    transport: oauth_transport.Provider,
    def: *const registry.Def,
) !?Started {
    if (active()) return null;

    const job_alloc = std.heap.c_allocator;
    const job = try job_alloc.create(Job);
    errdefer job_alloc.destroy(job);

    var started: Started = undefined;
    switch (def.oauth.style) {
        .oidc_device => {
            const pending = try oauth_flows.beginDeviceLogin(job_alloc, transport, def);
            job.* = .{ .transport = transport, .def = def, .pending = .{ .device = pending } };
            errdefer deinitJob(job_alloc, job);
            const notice = try std.fmt.allocPrint(
                alloc,
                "Open {s}\nCode: {s}\nWaiting for browser sign-in to {s}.",
                .{ pending.verification_url, pending.user_code, def.label },
            );
            const url_start = "Open ".len;
            started = .{
                .notice = notice,
                .browser_url = notice[url_start .. url_start + pending.verification_url.len],
            };
        },
        .pkce_loopback => {
            const pending = try oauth_flows.beginPkceLogin(job_alloc, def);
            job.* = .{ .transport = transport, .def = def, .pending = .{ .pkce = pending } };
            errdefer deinitJob(job_alloc, job);
            const notice = try std.fmt.allocPrint(
                alloc,
                "Complete the {s} sign-in in your browser:\n{s}",
                .{ def.label, pending.authorize_url },
            );
            started = .{
                .notice = notice,
                .browser_url = notice[notice.len - pending.authorize_url.len ..],
            };
        },
    }
    errdefer alloc.free(started.notice);

    active_def = def;
    storePhase(.waiting);
    const thread = std.Thread.spawn(.{}, runJob, .{job}) catch |err| {
        // Roll back so a later attempt can start; the pending state dies
        // with the job.
        storePhase(.idle);
        active_def = null;
        deinitJob(job_alloc, job);
        return err;
    };
    thread.detach();
    return started;
}

/// Reports a finished sign-in exactly once. The returned text is owned by
/// `alloc`.
pub fn takeOutcome(alloc: Allocator) !?Outcome {
    switch (loadPhase()) {
        .idle, .waiting => return null,
        .succeeded => {
            const def = active_def;
            active_def = null;
            storePhase(.idle);
            const label = if (def) |value| value.label else "the provider";
            const prefix = if (def) |value| value.model_prefix else "";
            return .{ .succeeded = try std.fmt.allocPrint(
                alloc,
                "Signed in to {s}. Pick a {s} model with /models to use it.",
                .{ label, prefix },
            ) };
        },
        .failed => {
            const def = active_def;
            active_def = null;
            const err_name = failure_name_buf[0..failure_name_len];
            const label = if (def) |value| value.label else "the provider";
            const body = try std.fmt.allocPrint(
                alloc,
                "Sign-in to {s} failed ({s}). Try again from /setup.",
                .{ label, err_name },
            );
            storePhase(.idle);
            return .{ .failed = body };
        },
    }
}

fn runJob(job: *Job) void {
    const alloc = std.heap.c_allocator;
    const result = switch (job.pending) {
        .device => |*pending| oauth_flows.awaitDeviceLogin(alloc, job.transport, job.def, pending),
        .pkce => |*pending| oauth_flows.awaitPkceLogin(alloc, job.transport, job.def, pending),
    };
    const provider_key = job.def.key;
    deinitJob(alloc, job);
    if (result) {
        debug_trace.logf("auth", "provider sign-in succeeded provider={s}", .{provider_key});
        storePhase(.succeeded);
    } else |err| {
        debug_trace.logf("auth", "provider sign-in failed provider={s} err={s}", .{ provider_key, @errorName(err) });
        const name = @errorName(err);
        const len = @min(name.len, max_failure_name_bytes);
        @memcpy(failure_name_buf[0..len], name[0..len]);
        failure_name_len = len;
        storePhase(.failed);
    }
}

fn deinitJob(alloc: Allocator, job: *Job) void {
    switch (job.pending) {
        .device => |*pending| pending.deinit(alloc),
        .pkce => |*pending| pending.deinit(alloc),
    }
    alloc.destroy(job);
}

test "sign-in starts idle and reports no outcome" {
    try std.testing.expect(!active());
    try std.testing.expectEqual(Phase.idle, loadPhase());
    try std.testing.expect((try takeOutcome(std.testing.allocator)) == null);
}

test "outcome text names the provider and its model prefix" {
    const alloc = std.testing.allocator;
    active_def = registry.byKey("openai").?;
    storePhase(.succeeded);
    const outcome = (try takeOutcome(alloc)).?;
    defer alloc.free(outcome.succeeded);
    try std.testing.expect(std.mem.find(u8, outcome.succeeded, "OpenAI") != null);
    try std.testing.expect(std.mem.find(u8, outcome.succeeded, "openai/") != null);
    try std.testing.expectEqual(Phase.idle, loadPhase());
    try std.testing.expect((try takeOutcome(alloc)) == null);
}

test "failure text carries the error name and clears the phase" {
    const alloc = std.testing.allocator;
    active_def = registry.byKey("xai").?;
    const name = "CallbackTimedOut";
    @memcpy(failure_name_buf[0..name.len], name);
    failure_name_len = name.len;
    storePhase(.failed);
    const outcome = (try takeOutcome(alloc)).?;
    defer alloc.free(outcome.failed);
    try std.testing.expect(std.mem.find(u8, outcome.failed, "xAI") != null);
    try std.testing.expect(std.mem.find(u8, outcome.failed, name) != null);
    try std.testing.expectEqual(Phase.idle, loadPhase());
}
