//! omfx fork-owned provider registry.
//!
//! Maps namespaced model ids ("openai/gpt-5.2") to direct provider
//! definitions so the router can serve them without the Vercel AI Gateway.
//! The gateway stays the default transport; a model routes directly only
//! when its provider has a live credential (see provider_credentials.zig).
//!
//! Adding a provider is a data change here plus a wire-protocol choice.

const std = @import("std");
const io_mod = @import("../shared/io.zig");

pub const WireProtocol = enum {
    /// OpenAI Responses API (`POST <base>/responses`).
    openai_responses,
    /// OpenAI Chat Completions API (`POST <base>/chat/completions`).
    openai_chat_completions,
};

pub const OAuthStyle = enum {
    /// Browser PKCE flow with a fixed loopback callback (OpenAI Codex style).
    pkce_loopback,
    /// RFC 8628 device authorization flow through OIDC discovery (xAI style).
    oidc_device,
};

pub const OAuthConfig = struct {
    style: OAuthStyle,
    issuer: []const u8,
    /// Authorization endpoint for the PKCE style; unused for device flow,
    /// which discovers endpoints from the issuer metadata.
    authorize_url: []const u8 = "",
    token_url: []const u8 = "",
    client_id: []const u8,
    scope: []const u8,
    callback_port: u16 = 0,
    callback_path: []const u8 = "",
};

pub const Def = struct {
    /// Stable key used in CLI arguments, session file names, and settings.
    key: []const u8,
    label: []const u8,
    /// CLI aliases accepted by `byKey` in addition to `key`.
    aliases: []const []const u8 = &.{},
    /// Namespace prefix that routes a model id to this provider.
    model_prefix: []const u8,
    /// Wire protocol and chat endpoint when authenticated with an API key.
    api_wire: WireProtocol,
    api_base_url: []const u8,
    api_chat_path: []const u8,
    /// Wire protocol and chat endpoint when authenticated with OAuth.
    oauth_wire: WireProtocol,
    oauth_base_url: []const u8,
    oauth_chat_path: []const u8,
    /// Environment variable holding a metered API key.
    api_key_env: []const u8,
    /// Fork-owned environment override for the base URL (tests, proxies).
    /// Applies to both API-key and OAuth transport when set.
    base_url_env: []const u8,
    oauth: OAuthConfig,
};

pub const defs = [_]Def{
    .{
        .key = "openai",
        .label = "OpenAI",
        .aliases = &.{"chatgpt"},
        .model_prefix = "openai/",
        .api_wire = .openai_responses,
        .api_base_url = "https://api.openai.com/v1",
        .api_chat_path = "/responses",
        .oauth_wire = .openai_responses,
        .oauth_base_url = "https://chatgpt.com/backend-api/codex",
        .oauth_chat_path = "/responses",
        .api_key_env = "OPENAI_API_KEY",
        .base_url_env = "OMFX_OPENAI_BASE_URL",
        .oauth = .{
            .style = .pkce_loopback,
            .issuer = "https://auth.openai.com",
            .authorize_url = "https://auth.openai.com/oauth/authorize",
            .token_url = "https://auth.openai.com/oauth/token",
            .client_id = "app_EMoamEEZ73f0CkXaXp7hrann",
            .scope = "openid profile email offline_access",
            .callback_port = 1455,
            .callback_path = "/auth/callback",
        },
    },
    .{
        .key = "xai",
        .label = "xAI",
        .aliases = &.{"grok"},
        .model_prefix = "xai/",
        .api_wire = .openai_chat_completions,
        .api_base_url = "https://api.x.ai/v1",
        .api_chat_path = "/chat/completions",
        .oauth_wire = .openai_chat_completions,
        .oauth_base_url = "https://api.x.ai/v1",
        .oauth_chat_path = "/chat/completions",
        .api_key_env = "XAI_API_KEY",
        .base_url_env = "OMFX_XAI_BASE_URL",
        .oauth = .{
            .style = .oidc_device,
            .issuer = "https://auth.x.ai",
            .client_id = "b1a00492-073a-47ea-816f-4c329264a828",
            .scope = "openid profile email offline_access grok-cli:access api:access",
        },
    },
};

pub fn byModel(model: []const u8) ?*const Def {
    for (&defs) |*def| {
        if (std.mem.startsWith(u8, model, def.model_prefix) and
            model.len > def.model_prefix.len) return def;
    }
    return null;
}

pub fn byKey(key: []const u8) ?*const Def {
    for (&defs) |*def| {
        if (std.ascii.eqlIgnoreCase(key, def.key)) return def;
        for (def.aliases) |alias| {
            if (std.ascii.eqlIgnoreCase(key, alias)) return def;
        }
    }
    return null;
}

/// The provider-native model id, with the routing namespace removed.
pub fn bareModel(def: *const Def, model: []const u8) []const u8 {
    if (std.mem.startsWith(u8, model, def.model_prefix)) {
        return model[def.model_prefix.len..];
    }
    return model;
}

/// Base URL for a transport kind, honoring the fork env override.
pub fn baseUrl(def: *const Def, oauth: bool) []const u8 {
    if (io_mod.getenv(def.base_url_env)) |override| {
        const trimmed = std.mem.trim(u8, override, " \t\r\n");
        if (trimmed.len > 0) return trimmed;
    }
    return if (oauth) def.oauth_base_url else def.api_base_url;
}

test "model routing matches only namespaced ids" {
    try std.testing.expectEqualStrings("openai", byModel("openai/gpt-5.2").?.key);
    try std.testing.expectEqualStrings("xai", byModel("xai/grok-4").?.key);
    try std.testing.expect(byModel("anthropic/claude-opus-4.8") == null);
    try std.testing.expect(byModel("openai/") == null);
    try std.testing.expect(byModel("openai") == null);
    try std.testing.expect(byModel("") == null);
}

test "provider keys resolve with aliases case-insensitively" {
    try std.testing.expectEqualStrings("xai", byKey("grok").?.key);
    try std.testing.expectEqualStrings("xai", byKey("XAI").?.key);
    try std.testing.expectEqualStrings("openai", byKey("OpenAI").?.key);
    try std.testing.expectEqualStrings("openai", byKey("chatgpt").?.key);
    try std.testing.expect(byKey("vercel") == null);
}

test "bare model strips only the owning namespace" {
    const openai = byKey("openai").?;
    try std.testing.expectEqualStrings("gpt-5.2-codex", bareModel(openai, "openai/gpt-5.2-codex"));
    try std.testing.expectEqualStrings("gpt-5.2", bareModel(openai, "gpt-5.2"));
}

test "base url falls back to transport-specific defaults" {
    const openai = byKey("openai").?;
    try std.testing.expectEqualStrings("https://api.openai.com/v1", baseUrl(openai, false));
    try std.testing.expectEqualStrings("https://chatgpt.com/backend-api/codex", baseUrl(openai, true));
    const xai = byKey("xai").?;
    try std.testing.expectEqualStrings("https://api.x.ai/v1", baseUrl(xai, true));
}
