//! omfx fork-owned per-provider credential resolution.
//!
//! A model routes directly to a provider only when one of these exists, in
//! precedence order: the provider's API-key environment variable, then a
//! stored OAuth session from `fx login <provider>`. Gateway credentials
//! (fx login with Vercel, AI_GATEWAY_API_KEY) are unrelated to this path.

const std = @import("std");
const oauth_transport = @import("../auth/oauth_transport.zig");
const secret = @import("../auth/secret.zig");
const io_mod = @import("../shared/io.zig");
const auth_store = @import("auth_store.zig");
const oauth_flows = @import("oauth_flows.zig");
const registry = @import("registry.zig");

const Allocator = std.mem.Allocator;

pub const Kind = enum {
    api_key,
    oauth,
};

pub const Credential = struct {
    kind: Kind,
    token: []u8,
    /// ChatGPT account id for the OpenAI subscription backend.
    account_id: ?[]u8 = null,

    pub fn deinit(self: *Credential, alloc: Allocator) void {
        secret.zeroAndFree(alloc, self.token);
        if (self.account_id) |account_id| alloc.free(account_id);
        self.* = undefined;
    }
};

fn envApiKey(def: *const registry.Def) ?[]const u8 {
    const raw = io_mod.getenv(def.api_key_env) orelse return null;
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) return null;
    return trimmed;
}

/// Cheap probe used on every routing decision; no parsing or refresh.
pub fn preferredKind(def: *const registry.Def) ?Kind {
    if (envApiKey(def) != null) return .api_key;
    if (auth_store.exists(def.key)) return .oauth;
    return null;
}

pub fn exists(def: *const registry.Def) bool {
    return preferredKind(def) != null;
}

/// True when any direct provider has an API key or valid stored OAuth session.
pub fn any_available(alloc: Allocator) !bool {
    for (&registry.defs) |*def| {
        if (envApiKey(def) != null) return true;
        var session = (try auth_store.load(alloc, def.key)) orelse continue;
        session.deinit(alloc);
        return true;
    }
    return false;
}

/// True when the model routes to a direct provider that has a credential,
/// which satisfies the app's credential gate without a gateway credential.
pub fn modelHasDirectCredential(model: []const u8) bool {
    const def = registry.byModel(model) orelse return false;
    return exists(def);
}

/// Status summary such as "openai (subscription), xai (api key)", or null
/// when no direct provider has a credential. The caller owns the result.
pub fn statusSummaryAlloc(alloc: Allocator) !?[]u8 {
    var out: std.Io.Writer.Allocating = .init(alloc);
    errdefer out.deinit();
    var any = false;
    for (&registry.defs) |*def| {
        const kind = preferredKind(def) orelse continue;
        if (any) try out.writer.writeAll(", ");
        try out.writer.print("{s} ({s})", .{ def.key, switch (kind) {
            .api_key => "api key",
            .oauth => "subscription",
        } });
        any = true;
    }
    if (!any) {
        out.deinit();
        return null;
    }
    return try out.toOwnedSlice();
}

/// Resolves the live credential, refreshing an expiring OAuth session.
/// The caller owns the returned credential.
pub fn resolve(
    alloc: Allocator,
    transport: oauth_transport.Provider,
    def: *const registry.Def,
) !Credential {
    if (envApiKey(def)) |key| {
        return .{ .kind = .api_key, .token = try alloc.dupe(u8, key) };
    }

    var session = (try auth_store.load(alloc, def.key)) orelse
        return error.NoProviderCredential;
    defer session.deinit(alloc);
    _ = try oauth_flows.refreshIfNeeded(alloc, transport, def, &session);

    const token = try alloc.dupe(u8, session.access_token);
    errdefer secret.zeroAndFree(alloc, token);
    const account_id = if (session.account_id) |id| try alloc.dupe(u8, id) else null;
    return .{ .kind = .oauth, .token = token, .account_id = account_id };
}

test "providers without any credential do not route" {
    var def = registry.byKey("openai").?.*;
    def.key = "omfx-test-nonexistent-provider";
    def.api_key_env = "OMFX_TEST_UNSET_API_KEY_ENV";
    try std.testing.expect(preferredKind(&def) == null);
    try std.testing.expect(!exists(&def));
}
