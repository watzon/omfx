//! omfx fork-owned OAuth login, refresh, and logout for direct providers.
//!
//! Upstream `core/auth/login_flow.zig` drives the single Vercel device flow.
//! The fork needs two more shapes, both described by `registry.OAuthConfig`:
//!
//! * `oidc_device` (xAI): OIDC discovery plus the RFC 8628 device flow. The
//!   issuer-generic parts come straight from `core/auth/oauth.zig`.
//! * `pkce_loopback` (OpenAI): a browser PKCE flow that returns the
//!   authorization code to a fixed loopback port.
//!
//! HTTP always goes through an injected `oauth_transport.Provider`, so every
//! flow below is exercised in tests without a socket. Persistence goes through
//! `auth_store`, which keeps one file per provider.

const std = @import("std");
const oauth = @import("../auth/oauth.zig");
const oauth_transport = @import("../auth/oauth_transport.zig");
const secret = @import("../auth/secret.zig");
const host = @import("../hosts/host.zig");
const host_target = @import("../hosts/target.zig");
const url_opener = @import("../hosts/url_opener.zig");
const debug_trace = @import("../shared/debug_trace.zig");
const io_mod = @import("../shared/io.zig");
const auth_store = @import("auth_store.zig");
const registry = @import("registry.zig");

const Allocator = std.mem.Allocator;

/// Bound on any single OAuth response body. Token endpoints answer in
/// hundreds of bytes; anything larger is a redirect or an error page.
const max_response_bytes: usize = 256 * 1024;
const max_jwt_payload_bytes: usize = 64 * 1024;
const max_request_line_bytes: usize = 8 * 1024;
const request_timeout_ms: u64 = 30 * std.time.ms_per_s;
const max_poll_interval_ms: u64 = 60 * std.time.ms_per_s;
const slow_down_step_ms: u64 = 5 * std.time.ms_per_s;
const callback_deadline_ms: u64 = 5 * 60 * std.time.ms_per_s;
const default_expires_in_s: i64 = 3600;

/// The claim OpenAI puts the ChatGPT account id under, in both the id token
/// and the access token.
const openai_auth_claim = "https://api.openai.com/auth";
/// The Codex CLI narrows the scope on refresh; the issuer rejects a widened one.
const openai_refresh_scope = "openid profile email";

const callback_success_body =
    "<!doctype html><html><body><p>Sign-in complete. " ++
    "You can close this tab and return to the terminal.</p></body></html>";

pub const FlowError = error{
    /// The stored refresh token is no longer accepted. The user must sign in again.
    ReauthenticationRequired,
    /// Another process holds the fixed loopback callback port.
    CallbackPortBusy,
    CallbackTimedOut,
    CallbackStateMismatch,
    InvalidCallbackRequest,
    AuthorizationDenied,
    DeviceCodeExpired,
    InvalidTokenResponse,
    /// A discovered endpoint left the issuer's registrable domain, or is not HTTPS.
    UntrustedOAuthEndpoint,
    /// Direct provider sign-in is native-only.
    DirectProviderAuthUnsupported,
};

pub const LoginOutcome = struct {
    provider_label: []const u8,
    account_id_present: bool,
};

/// Signs in to `def` interactively, printing progress to stderr, and stores
/// the resulting session. Dispatches on `def.oauth.style`.
pub fn runLogin(
    alloc: Allocator,
    transport: oauth_transport.Provider,
    def: *const registry.Def,
) !void {
    _ = try runLoginWith(alloc, transport, def, .{});
}

/// Refreshes the session when it is inside the 60 second expiry skew, then
/// persists it. Returns true when a refresh happened.
pub fn refreshIfNeeded(
    alloc: Allocator,
    transport: oauth_transport.Provider,
    def: *const registry.Def,
    session: *auth_store.Session,
) !bool {
    return refreshIfNeededWith(alloc, transport, def, session, .{});
}

/// Removes the stored session. Returns true when one was removed.
pub fn logout(alloc: Allocator, def: *const registry.Def) !bool {
    _ = alloc;
    return auth_store.delete(def.key);
}

// --- Injected dependencies ----------------------------------------------
//
// Production uses the defaults. Tests replace the clock, the sleeper, the
// terminal, the browser, and the session store, which is what makes the
// flows below testable without a network or a home directory.

const Store = struct {
    ctx: ?*anyopaque = null,
    save_fn: *const fn (?*anyopaque, Allocator, auth_store.Session) anyerror!void = defaultSave,
};

fn defaultSave(_: ?*anyopaque, alloc: Allocator, session: auth_store.Session) anyerror!void {
    return auth_store.save(alloc, session);
}

const Deps = struct {
    ctx: ?*anyopaque = null,
    now_ms: *const fn (?*anyopaque) i64 = defaultNowMs,
    sleep_ms: *const fn (?*anyopaque, u64) void = defaultSleepMs,
    notify_fn: *const fn (?*anyopaque, []const u8) void = defaultNotify,
    open_url_fn: *const fn (?*anyopaque, Allocator, []const u8) bool = defaultOpenUrl,
    store: Store = .{},

    fn now(self: Deps) i64 {
        return self.now_ms(self.ctx);
    }

    fn sleep(self: Deps, millis: u64) void {
        self.sleep_ms(self.ctx, millis);
    }

    fn say(self: Deps, text: []const u8) void {
        self.notify_fn(self.ctx, text);
    }

    fn sayFmt(self: Deps, comptime fmt: []const u8, args: anytype) void {
        var buf: [1024]u8 = undefined;
        const text = std.fmt.bufPrint(&buf, fmt, args) catch return;
        self.say(text);
    }

    fn openUrl(self: Deps, alloc: Allocator, url: []const u8) bool {
        return self.open_url_fn(self.ctx, alloc, url);
    }

    fn save(self: Deps, alloc: Allocator, session: auth_store.Session) !void {
        return self.store.save_fn(self.store.ctx, alloc, session);
    }
};

fn defaultNowMs(_: ?*anyopaque) i64 {
    return io_mod.milliTimestamp();
}

fn defaultSleepMs(_: ?*anyopaque, millis: u64) void {
    io_mod.sleep(millis *| std.time.ns_per_ms);
}

fn defaultNotify(_: ?*anyopaque, text: []const u8) void {
    std.Io.File.stderr().writeStreamingAll(io_mod.getIo(), text) catch {};
}

/// Mirrors `login_flow`: `FX_NO_OPEN_BROWSER` suppresses the launch, and the
/// URL is always printed so a manual copy still works.
fn defaultOpenUrl(_: ?*anyopaque, alloc: Allocator, url: []const u8) bool {
    if (io_mod.getenv("FX_NO_OPEN_BROWSER") != null) return false;
    if (!host.current().native_url_open) return false;
    return url_opener.native_opener.open(alloc, url) catch false;
}

// --- Login dispatch ------------------------------------------------------

fn runLoginWith(
    alloc: Allocator,
    transport: oauth_transport.Provider,
    def: *const registry.Def,
    deps: Deps,
) !LoginOutcome {
    const outcome = switch (def.oauth.style) {
        .oidc_device => try runDeviceLogin(alloc, transport, def, deps),
        .pkce_loopback => try runPkceLogin(alloc, transport, def, deps),
    };
    deps.sayFmt("Signed in to {s}.\n", .{outcome.provider_label});
    return outcome;
}

// --- oidc_device (xAI) ---------------------------------------------------

fn runDeviceLogin(
    alloc: Allocator,
    transport: oauth_transport.Provider,
    def: *const registry.Def,
    deps: Deps,
) !LoginOutcome {
    var metadata = try oauth.discover(alloc, transport, def.oauth.issuer);
    defer metadata.deinit(alloc);
    try validateIssuerEndpoint(def.oauth.issuer, metadata.device_authorization_endpoint);
    try validateIssuerEndpoint(def.oauth.issuer, metadata.token_endpoint);

    var device = try requestDeviceAuthorization(
        alloc,
        transport,
        metadata.device_authorization_endpoint,
        def.oauth.client_id,
        def.oauth.scope,
    );
    defer device.deinit(alloc);

    const display_url = device.verification_uri_complete orelse device.verification_uri;
    deps.sayFmt("Open {s}\nCode: {s}\n\n", .{ display_url, device.user_code });
    _ = deps.openUrl(alloc, display_url);
    deps.say("Waiting for authorization...\n");

    var token = try pollForDeviceToken(alloc, transport, metadata, def, device, deps);
    defer token.deinit(alloc);

    const expires_at_ms = try oauth.expiry_timestamp_ms(deps.now(), token.expires_in);
    var session = try ownSession(alloc, .{
        .provider = def.key,
        .access_token = token.access_token,
        .refresh_token = token.refresh_token,
        .expires_at_ms = expires_at_ms,
        .token_url = metadata.token_endpoint,
        .client_id = def.oauth.client_id,
        .scope = if (token.scope.len > 0) token.scope else def.oauth.scope,
        .account_id = null,
    });
    defer session.deinit(alloc);
    try deps.save(alloc, session);

    return .{ .provider_label = def.label, .account_id_present = false };
}

/// The upstream helper hard-codes the Vercel scope, so the fork owns the form.
/// Parsing stays with `oauth.parseDeviceAuthorization`, which is issuer-generic.
fn requestDeviceAuthorization(
    alloc: Allocator,
    transport: oauth_transport.Provider,
    endpoint: []const u8,
    client_id: []const u8,
    scope: []const u8,
) !oauth.DeviceAuthorization {
    var form: FormBody = .{};
    var body: std.Io.Writer.Allocating = .init(alloc);
    defer body.deinit();
    try form.append(&body.writer, "client_id", client_id);
    try form.append(&body.writer, "scope", scope);

    const bytes = try fetchJson(alloc, transport, .post_form, endpoint, body.written());
    defer secret.zeroAndFree(alloc, bytes);
    return oauth.parseDeviceAuthorization(alloc, bytes);
}

fn pollForDeviceToken(
    alloc: Allocator,
    transport: oauth_transport.Provider,
    metadata: oauth.Metadata,
    def: *const registry.Def,
    device: oauth.DeviceAuthorization,
    deps: Deps,
) !oauth.TokenSet {
    var interval_ms = pollIntervalMs(device.interval);
    const started_ms = deps.now();
    const window_ms: i64 = @max(device.expires_in, 1) *| std.time.ms_per_s;
    var cancel_flag = std.atomic.Value(bool).init(false);

    while (true) {
        if (deps.now() -| started_ms > window_ms) {
            deps.say("The sign-in code expired before it was approved.\n");
            return FlowError.DeviceCodeExpired;
        }
        deps.sleep(interval_ms);

        const deadline = std.Io.Clock.Timestamp.fromNow(io_mod.getIo(), .{
            .clock = .awake,
            .raw = .fromMilliseconds(@intCast(request_timeout_ms)),
        });
        const result = oauth.pollDeviceTokenBounded(
            alloc,
            transport,
            metadata,
            def.oauth.client_id,
            device.device_code,
            &cancel_flag,
            deadline,
        ) catch |err| {
            switch (err) {
                oauth.OAuthError.AccessDenied => deps.say("Sign-in was denied in the browser.\n"),
                oauth.OAuthError.ExpiredToken => deps.say("The sign-in code expired before it was approved.\n"),
                else => {},
            }
            debug_trace.logf(
                "auth",
                "provider device poll failed provider={s} err={s}",
                .{ def.key, @errorName(err) },
            );
            return err;
        };
        switch (result) {
            .success => |token| return token,
            .pending => {},
            .slow_down => {
                interval_ms +|= slow_down_step_ms;
                if (interval_ms > max_poll_interval_ms) return oauth.OAuthError.InvalidOAuthResponse;
            },
        }
    }
}

fn pollIntervalMs(interval_seconds: i64) u64 {
    const seconds: u64 = @intCast(@max(interval_seconds, 1));
    return @min(seconds *| std.time.ms_per_s, max_poll_interval_ms);
}

// --- pkce_loopback (OpenAI) ---------------------------------------------

fn runPkceLogin(
    alloc: Allocator,
    transport: oauth_transport.Provider,
    def: *const registry.Def,
    deps: Deps,
) !LoginOutcome {
    var pkce = generatePkce();
    defer pkce.zero();
    const state = randomStateHex();

    var listener = Loopback.bind(def.oauth.callback_port) catch |err| switch (err) {
        FlowError.CallbackPortBusy => {
            deps.sayFmt(
                "Port {d} is already in use. {s} only accepts the sign-in redirect on that " ++
                    "exact port, so close the other listener and run the sign-in again.\n",
                .{ def.oauth.callback_port, def.label },
            );
            return err;
        },
        else => return err,
    };
    defer listener.deinit();

    const redirect_uri = try std.fmt.allocPrint(alloc, "http://localhost:{d}{s}", .{
        def.oauth.callback_port,
        def.oauth.callback_path,
    });
    defer alloc.free(redirect_uri);

    const authorize_url = try buildAuthorizeUrl(alloc, def, redirect_uri, &pkce.challenge, &state);
    defer alloc.free(authorize_url);

    deps.sayFmt("Open {s}\n\n", .{authorize_url});
    _ = deps.openUrl(alloc, authorize_url);
    deps.say("Waiting for the browser to finish sign-in. Press ctrl+c to cancel.\n");

    var callback = try listener.awaitCallback(alloc, callback_deadline_ms);
    defer callback.deinit(alloc);

    if (callback.err_code) |code| {
        deps.sayFmt("Sign-in failed: {s}.\n", .{callback.err_description orelse code});
        return FlowError.AuthorizationDenied;
    }
    const returned_state = callback.state orelse return FlowError.CallbackStateMismatch;
    if (!std.mem.eql(u8, returned_state, &state)) return FlowError.CallbackStateMismatch;
    const code = callback.code orelse return FlowError.InvalidCallbackRequest;

    var tokens = try exchangeAuthorizationCode(
        alloc,
        transport,
        def,
        code,
        redirect_uri,
        &pkce.verifier,
    );
    defer tokens.deinit(alloc);

    const account_id = accountIdFromTokens(alloc, tokens);
    defer if (account_id) |value| alloc.free(value);

    const expires_at_ms = try oauth.expiry_timestamp_ms(deps.now(), tokens.expires_in);
    var session = try ownSession(alloc, .{
        .provider = def.key,
        .access_token = tokens.access_token,
        .refresh_token = tokens.refresh_token,
        .expires_at_ms = expires_at_ms,
        .token_url = def.oauth.token_url,
        .client_id = def.oauth.client_id,
        .scope = tokens.scope orelse def.oauth.scope,
        .account_id = account_id,
    });
    defer session.deinit(alloc);
    try deps.save(alloc, session);

    if (account_id == null) {
        deps.say("Signed in, but the account id was missing from the token response.\n");
    }
    return .{ .provider_label = def.label, .account_id_present = account_id != null };
}

fn exchangeAuthorizationCode(
    alloc: Allocator,
    transport: oauth_transport.Provider,
    def: *const registry.Def,
    code: []const u8,
    redirect_uri: []const u8,
    verifier: []const u8,
) !Tokens {
    var form: FormBody = .{};
    var body: std.Io.Writer.Allocating = .init(alloc);
    defer body.deinit();
    try form.append(&body.writer, "grant_type", "authorization_code");
    try form.append(&body.writer, "code", code);
    try form.append(&body.writer, "redirect_uri", redirect_uri);
    try form.append(&body.writer, "client_id", def.oauth.client_id);
    try form.append(&body.writer, "code_verifier", verifier);

    const bytes = try fetchJson(alloc, transport, .post_form, def.oauth.token_url, body.written());
    defer secret.zeroAndFree(alloc, bytes);
    return parseTokens(alloc, bytes);
}

fn buildAuthorizeUrl(
    alloc: Allocator,
    def: *const registry.Def,
    redirect_uri: []const u8,
    challenge: []const u8,
    state: []const u8,
) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(alloc);
    errdefer out.deinit();
    const writer = &out.writer;
    try writer.writeAll(def.oauth.authorize_url);
    try writer.writeByte('?');

    var query: FormBody = .{};
    try query.append(writer, "response_type", "code");
    try query.append(writer, "client_id", def.oauth.client_id);
    try query.append(writer, "redirect_uri", redirect_uri);
    try query.append(writer, "scope", def.oauth.scope);
    try query.append(writer, "code_challenge", challenge);
    try query.append(writer, "code_challenge_method", "S256");
    try query.append(writer, "state", state);
    // Both flags come from the Codex CLI authorize request. The issuer relies
    // on them to pick the simplified consent screen.
    try query.append(writer, "id_token_add_organizations", "false");
    try query.append(writer, "codex_cli_simplified_flow", "true");
    return out.toOwnedSlice();
}

// --- PKCE ---------------------------------------------------------------

const b64 = std.base64.url_safe_no_pad;
const verifier_len = 86; // base64url, no padding, of 64 random bytes
const challenge_len = 43; // base64url, no padding, of a SHA-256 digest

const Pkce = struct {
    verifier: [verifier_len]u8,
    challenge: [challenge_len]u8,

    fn zero(self: *Pkce) void {
        std.crypto.secureZero(u8, &self.verifier);
    }
};

fn generatePkce() Pkce {
    var raw: [64]u8 = undefined;
    io_mod.getIo().random(&raw);
    defer std.crypto.secureZero(u8, &raw);

    var pkce: Pkce = undefined;
    _ = b64.Encoder.encode(&pkce.verifier, &raw);
    pkce.challenge = pkceChallenge(&pkce.verifier);
    return pkce;
}

fn pkceChallenge(verifier: []const u8) [challenge_len]u8 {
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(verifier, &digest, .{});
    var out: [challenge_len]u8 = undefined;
    _ = b64.Encoder.encode(&out, &digest);
    return out;
}

fn randomStateHex() [32]u8 {
    var raw: [16]u8 = undefined;
    io_mod.getIo().random(&raw);
    return std.fmt.bytesToHex(raw, .lower);
}

// --- Loopback callback listener -----------------------------------------

const Callback = struct {
    code: ?[]u8 = null,
    state: ?[]u8 = null,
    err_code: ?[]u8 = null,
    err_description: ?[]u8 = null,

    fn deinit(self: *Callback, alloc: Allocator) void {
        if (self.code) |value| secret.zeroAndFree(alloc, value);
        if (self.state) |value| alloc.free(value);
        if (self.err_code) |value| alloc.free(value);
        if (self.err_description) |value| alloc.free(value);
        self.* = .{};
    }
};

const Loopback = if (host_target.is_wasm) UnsupportedLoopback else NativeLoopback;

const UnsupportedLoopback = struct {
    fn bind(_: u16) !UnsupportedLoopback {
        return FlowError.DirectProviderAuthUnsupported;
    }

    fn deinit(_: *UnsupportedLoopback) void {}

    fn port(_: *UnsupportedLoopback) u16 {
        return 0;
    }

    fn awaitCallback(_: *UnsupportedLoopback, _: Allocator, _: u64) !Callback {
        return FlowError.DirectProviderAuthUnsupported;
    }
};

const NativeLoopback = struct {
    server: std.Io.net.Server,

    /// Binds 127.0.0.1 on the exact registered redirect port. `reuse_address`
    /// stays off on purpose: the flow must notice a competing listener rather
    /// than share the port with it.
    fn bind(callback_port: u16) !NativeLoopback {
        var address = try std.Io.net.IpAddress.parse("127.0.0.1", callback_port);
        const server = address.listen(io_mod.getIo(), .{ .reuse_address = false }) catch |err| switch (err) {
            error.AddressInUse => return FlowError.CallbackPortBusy,
            else => return err,
        };
        return .{ .server = server };
    }

    fn deinit(self: *NativeLoopback) void {
        self.server.deinit(io_mod.getIo());
    }

    fn port(self: *NativeLoopback) u16 {
        return self.server.socket.address.getPort();
    }

    /// Accepts one request, answers it, and closes. The bound wait is a poll
    /// on the listening socket, the same technique `core/terminal/host.zig`
    /// uses, because `std.Io.net.Server.accept` has no deadline of its own.
    fn awaitCallback(self: *NativeLoopback, alloc: Allocator, deadline_ms: u64) !Callback {
        const zio = io_mod.getIo();
        var stream = try self.accept(deadline_ms);
        defer stream.close(zio);

        var read_buffer: [max_request_line_bytes]u8 = undefined;
        var reader = stream.reader(zio, &read_buffer);
        const line = reader.interface.takeDelimiterExclusive('\n') catch {
            return FlowError.InvalidCallbackRequest;
        };

        var callback = parseCallbackRequestLine(alloc, std.mem.trimEnd(u8, line, "\r")) catch |err| {
            respond(stream, "Sign-in failed. Return to the terminal.");
            return err;
        };
        errdefer callback.deinit(alloc);

        if (callback.err_description orelse callback.err_code) |message| {
            var body_buf: [512]u8 = undefined;
            const body = std.fmt.bufPrint(
                &body_buf,
                "<!doctype html><html><body><p>Sign-in failed: {s}</p></body></html>",
                .{message},
            ) catch "<!doctype html><html><body><p>Sign-in failed.</p></body></html>";
            respond(stream, body);
        } else {
            respond(stream, callback_success_body);
        }
        return callback;
    }

    fn accept(self: *NativeLoopback, deadline_ms: u64) !std.Io.net.Stream {
        var remaining_ms = deadline_ms;
        while (true) {
            var poll_fds = [_]std.posix.pollfd{.{
                .fd = self.server.socket.handle,
                .events = std.posix.POLL.IN,
                .revents = 0,
            }};
            const slice_ms: i32 = @intCast(@min(remaining_ms, 250));
            const ready = std.posix.poll(&poll_fds, slice_ms) catch return FlowError.CallbackTimedOut;
            if (ready > 0 and (poll_fds[0].revents & std.posix.POLL.IN) != 0) {
                return self.server.accept(io_mod.getIo());
            }
            const waited: u64 = @intCast(slice_ms);
            if (remaining_ms <= waited) return FlowError.CallbackTimedOut;
            remaining_ms -= waited;
        }
    }

    fn respond(stream: std.Io.net.Stream, body: []const u8) void {
        var write_buffer: [1024]u8 = undefined;
        var writer = stream.writer(io_mod.getIo(), &write_buffer);
        writer.interface.print(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n" ++
                "Content-Length: {d}\r\nConnection: close\r\n\r\n{s}",
            .{ body.len, body },
        ) catch {};
        writer.interface.flush() catch {};
    }
};

/// Pure parser for the callback request line, for example
/// `GET /auth/callback?code=abc&state=xyz HTTP/1.1`.
fn parseCallbackRequestLine(alloc: Allocator, line: []const u8) !Callback {
    if (!std.mem.startsWith(u8, line, "GET ")) return FlowError.InvalidCallbackRequest;
    const rest = line[4..];
    const target = rest[0 .. std.mem.findScalar(u8, rest, ' ') orelse rest.len];
    const question = std.mem.findScalar(u8, target, '?') orelse return FlowError.InvalidCallbackRequest;

    var callback: Callback = .{};
    errdefer callback.deinit(alloc);

    var pairs = std.mem.tokenizeScalar(u8, target[question + 1 ..], '&');
    while (pairs.next()) |pair| {
        const equals = std.mem.findScalar(u8, pair, '=') orelse continue;
        const key = pair[0..equals];
        const slot: *?[]u8 = if (std.mem.eql(u8, key, "code"))
            &callback.code
        else if (std.mem.eql(u8, key, "state"))
            &callback.state
        else if (std.mem.eql(u8, key, "error"))
            &callback.err_code
        else if (std.mem.eql(u8, key, "error_description"))
            &callback.err_description
        else
            continue;
        if (slot.* != null) continue;
        slot.* = try percentDecodeAlloc(alloc, pair[equals + 1 ..]);
    }
    return callback;
}

fn percentDecodeAlloc(alloc: Allocator, value: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    var index: usize = 0;
    while (index < value.len) {
        const byte = value[index];
        if (byte == '+') {
            try out.append(alloc, ' ');
            index += 1;
        } else if (byte == '%' and index + 2 < value.len) {
            const high = std.fmt.charToDigit(value[index + 1], 16) catch
                return FlowError.InvalidCallbackRequest;
            const low = std.fmt.charToDigit(value[index + 2], 16) catch
                return FlowError.InvalidCallbackRequest;
            try out.append(alloc, (@as(u8, high) << 4) | @as(u8, low));
            index += 3;
        } else {
            try out.append(alloc, byte);
            index += 1;
        }
    }
    return out.toOwnedSlice(alloc);
}

// --- Refresh -------------------------------------------------------------

fn refreshIfNeededWith(
    alloc: Allocator,
    transport: oauth_transport.Provider,
    def: *const registry.Def,
    session: *auth_store.Session,
    deps: Deps,
) !bool {
    if (deps.now() < auth_store.refreshDeadlineMs(session.expires_at_ms)) return false;
    const refresh_token = session.refresh_token orelse return FlowError.ReauthenticationRequired;

    var form: FormBody = .{};
    var body: std.Io.Writer.Allocating = .init(alloc);
    defer body.deinit();
    try form.append(&body.writer, "grant_type", "refresh_token");
    try form.append(&body.writer, "refresh_token", refresh_token);
    try form.append(&body.writer, "client_id", def.oauth.client_id);
    if (def.oauth.style == .pkce_loopback) {
        try form.append(&body.writer, "scope", openai_refresh_scope);
    }

    const bytes = try fetchJson(alloc, transport, .post_form, session.token_url, body.written());
    defer secret.zeroAndFree(alloc, bytes);
    var tokens = try parseTokens(alloc, bytes);
    defer tokens.deinit(alloc);

    const expires_at_ms = try oauth.expiry_timestamp_ms(deps.now(), tokens.expires_in);

    const access_token = try alloc.dupe(u8, tokens.access_token);
    errdefer secret.zeroAndFree(alloc, access_token);
    // xAI rotates refresh tokens; OpenAI usually replays the existing one.
    const rotated: ?[]u8 = if (tokens.refresh_token) |value| try alloc.dupe(u8, value) else null;
    errdefer if (rotated) |value| secret.zeroAndFree(alloc, value);
    const account_id = accountIdFromTokens(alloc, tokens);
    errdefer if (account_id) |value| alloc.free(value);

    secret.zeroAndFree(alloc, session.access_token);
    session.access_token = access_token;
    if (rotated) |value| {
        if (session.refresh_token) |previous| secret.zeroAndFree(alloc, previous);
        session.refresh_token = value;
    }
    if (account_id) |value| {
        if (session.account_id) |previous| alloc.free(previous);
        session.account_id = value;
    }
    session.expires_at_ms = expires_at_ms;

    try deps.save(alloc, session.*);
    debug_trace.logf("auth", "provider session refreshed provider={s}", .{def.key});
    return true;
}

// --- Token responses -----------------------------------------------------

const Tokens = struct {
    access_token: []u8,
    refresh_token: ?[]u8 = null,
    id_token: ?[]u8 = null,
    scope: ?[]u8 = null,
    expires_in: i64,

    fn deinit(self: *Tokens, alloc: Allocator) void {
        secret.zeroAndFree(alloc, self.access_token);
        if (self.refresh_token) |value| secret.zeroAndFree(alloc, value);
        if (self.id_token) |value| secret.zeroAndFree(alloc, value);
        if (self.scope) |value| alloc.free(value);
        self.* = undefined;
    }
};

/// `oauth.parseTokenSet` cannot serve the fork here: it requires `token_type`
/// and `expires_in`, and it drops `id_token`, which is the only place the
/// ChatGPT account id appears.
fn parseTokens(alloc: Allocator, bytes: []const u8) !Tokens {
    var parsed = try std.json.parseFromSlice(std.json.Value, alloc, bytes, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return FlowError.InvalidTokenResponse;
    const object = parsed.value.object;

    if (object.get("token_type")) |token_type| {
        if (token_type != .string or !std.ascii.eqlIgnoreCase(token_type.string, "Bearer")) {
            return FlowError.InvalidTokenResponse;
        }
    }

    const access_token = try dupeRequiredString(alloc, object, "access_token");
    errdefer secret.zeroAndFree(alloc, access_token);
    const refresh_token = try dupeOptionalString(alloc, object, "refresh_token");
    errdefer if (refresh_token) |value| secret.zeroAndFree(alloc, value);
    const id_token = try dupeOptionalString(alloc, object, "id_token");
    errdefer if (id_token) |value| secret.zeroAndFree(alloc, value);
    const scope = try dupeOptionalString(alloc, object, "scope");
    errdefer if (scope) |value| alloc.free(value);

    const expires_in = blk: {
        const value = object.get("expires_in") orelse break :blk default_expires_in_s;
        if (value == .null) break :blk default_expires_in_s;
        if (value != .integer or value.integer <= 0) return FlowError.InvalidTokenResponse;
        break :blk value.integer;
    };

    return .{
        .access_token = access_token,
        .refresh_token = refresh_token,
        .id_token = id_token,
        .scope = scope,
        .expires_in = expires_in,
    };
}

/// Prefers the id token, then falls back to the access token, because the
/// issuer puts the same claim in both.
fn accountIdFromTokens(alloc: Allocator, tokens: Tokens) ?[]u8 {
    if (tokens.id_token) |id_token| {
        if (accountIdFromJwt(alloc, id_token)) |value| return value;
    }
    return accountIdFromJwt(alloc, tokens.access_token);
}

fn accountIdFromJwt(alloc: Allocator, token: []const u8) ?[]u8 {
    var segments = std.mem.splitScalar(u8, token, '.');
    _ = segments.next() orelse return null;
    const encoded = segments.next() orelse return null;
    if (encoded.len == 0) return null;

    const size = b64.Decoder.calcSizeForSlice(encoded) catch return null;
    if (size == 0 or size > max_jwt_payload_bytes) return null;
    const payload = alloc.alloc(u8, size) catch return null;
    defer alloc.free(payload);
    b64.Decoder.decode(payload, encoded) catch return null;

    var parsed = std.json.parseFromSlice(std.json.Value, alloc, payload, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const claim = parsed.value.object.get(openai_auth_claim) orelse return null;
    if (claim != .object) return null;
    const account_id = claim.object.get("chatgpt_account_id") orelse return null;
    if (account_id != .string or account_id.string.len == 0) return null;
    return alloc.dupe(u8, account_id.string) catch null;
}

// --- HTTP ----------------------------------------------------------------

fn fetchJson(
    alloc: Allocator,
    transport: oauth_transport.Provider,
    method: oauth_transport.Method,
    url: []const u8,
    payload: ?[]const u8,
) ![]u8 {
    var response = try transport.execute(alloc, .{
        .method = method,
        .url = url,
        .payload = payload,
    });
    defer response.deinit(alloc);
    if (response.body.len > max_response_bytes) return oauth.OAuthError.InvalidOAuthResponse;
    if (response.disposition == .accepted) return response.takeBody();
    return mapErrorBody(alloc, response.body);
}

fn mapErrorBody(alloc: Allocator, body: []const u8) anyerror {
    var parsed = std.json.parseFromSlice(std.json.Value, alloc, body, .{}) catch
        return oauth.OAuthError.OAuthRequestFailed;
    defer parsed.deinit();
    if (parsed.value != .object) return oauth.OAuthError.OAuthRequestFailed;
    const value = parsed.value.object.get("error") orelse return oauth.OAuthError.OAuthRequestFailed;
    if (value != .string) return oauth.OAuthError.OAuthRequestFailed;
    const code = value.string;
    if (std.mem.eql(u8, code, "authorization_pending")) return oauth.OAuthError.AuthorizationPending;
    if (std.mem.eql(u8, code, "slow_down")) return oauth.OAuthError.SlowDown;
    if (std.mem.eql(u8, code, "access_denied")) return oauth.OAuthError.AccessDenied;
    if (std.mem.eql(u8, code, "expired_token")) return oauth.OAuthError.ExpiredToken;
    if (std.mem.eql(u8, code, "invalid_client")) return oauth.OAuthError.InvalidClient;
    // A rejected refresh token is the one error the caller must turn into a
    // fresh interactive sign-in.
    if (std.mem.eql(u8, code, "invalid_grant")) return FlowError.ReauthenticationRequired;
    return oauth.OAuthError.OAuthRequestFailed;
}

const FormBody = struct {
    first: bool = true,

    fn append(self: *FormBody, writer: *std.Io.Writer, key: []const u8, value: []const u8) !void {
        if (!self.first) try writer.writeAll("&");
        self.first = false;
        try percentEncode(writer, key);
        try writer.writeAll("=");
        try percentEncode(writer, value);
    }
};

fn percentEncode(writer: *std.Io.Writer, value: []const u8) !void {
    const hex = "0123456789ABCDEF";
    for (value) |byte| {
        const safe = std.ascii.isAlphanumeric(byte) or
            byte == '-' or byte == '_' or byte == '.' or byte == '~';
        if (safe) {
            try writer.writeByte(byte);
        } else {
            try writer.writeByte('%');
            try writer.writeByte(hex[byte >> 4]);
            try writer.writeByte(hex[byte & 0x0f]);
        }
    }
}

// --- Endpoint trust ------------------------------------------------------

/// Discovery documents are attacker-reachable input. An endpoint is accepted
/// only when it is HTTPS, carries no credentials, and stays inside the
/// issuer's registrable domain, meaning the last two labels of the issuer
/// host. For `https://auth.x.ai` that admits `x.ai` and any `*.x.ai` host.
fn validateIssuerEndpoint(issuer_url: []const u8, endpoint: []const u8) !void {
    const issuer_uri = std.Uri.parse(issuer_url) catch return FlowError.UntrustedOAuthEndpoint;
    const endpoint_uri = std.Uri.parse(endpoint) catch return FlowError.UntrustedOAuthEndpoint;
    if (!std.ascii.eqlIgnoreCase(endpoint_uri.scheme, "https")) {
        return FlowError.UntrustedOAuthEndpoint;
    }
    if (endpoint_uri.user != null or endpoint_uri.password != null) {
        return FlowError.UntrustedOAuthEndpoint;
    }

    var issuer_buf: [std.Io.net.HostName.max_len]u8 = undefined;
    var endpoint_buf: [std.Io.net.HostName.max_len]u8 = undefined;
    const issuer_component = issuer_uri.host orelse return FlowError.UntrustedOAuthEndpoint;
    const endpoint_component = endpoint_uri.host orelse return FlowError.UntrustedOAuthEndpoint;
    const issuer_host = issuer_component.toRaw(&issuer_buf) catch return FlowError.UntrustedOAuthEndpoint;
    const endpoint_host = endpoint_component.toRaw(&endpoint_buf) catch return FlowError.UntrustedOAuthEndpoint;

    if (!hostWithinRegistrableDomain(issuer_host, endpoint_host)) {
        return FlowError.UntrustedOAuthEndpoint;
    }
}

fn hostWithinRegistrableDomain(issuer_host: []const u8, endpoint_host: []const u8) bool {
    const domain = registrableDomain(issuer_host) orelse return false;
    if (std.ascii.eqlIgnoreCase(endpoint_host, domain)) return true;
    if (endpoint_host.len <= domain.len + 1) return false;
    const boundary = endpoint_host.len - domain.len - 1;
    return endpoint_host[boundary] == '.' and
        std.ascii.eqlIgnoreCase(endpoint_host[boundary + 1 ..], domain);
}

/// The last two labels of a host. Returns null for a bare label or an
/// address literal, which never carry a registrable domain.
fn registrableDomain(hostname: []const u8) ?[]const u8 {
    if (hostname.len == 0) return null;
    const last_dot = std.mem.lastIndexOfScalar(u8, hostname, '.') orelse return null;
    if (last_dot + 1 == hostname.len) return null;
    const previous_dot = std.mem.lastIndexOfScalar(u8, hostname[0..last_dot], '.');
    const start = if (previous_dot) |index| index + 1 else 0;
    return hostname[start..];
}

// --- Session construction ------------------------------------------------

const SessionInput = struct {
    provider: []const u8,
    access_token: []const u8,
    refresh_token: ?[]const u8 = null,
    expires_at_ms: i64,
    token_url: []const u8,
    client_id: []const u8,
    scope: ?[]const u8 = null,
    account_id: ?[]const u8 = null,
};

fn ownSession(alloc: Allocator, input: SessionInput) !auth_store.Session {
    const provider = try alloc.dupe(u8, input.provider);
    errdefer alloc.free(provider);
    const access_token = try alloc.dupe(u8, input.access_token);
    errdefer secret.zeroAndFree(alloc, access_token);
    const refresh_token: ?[]u8 = if (input.refresh_token) |value| try alloc.dupe(u8, value) else null;
    errdefer if (refresh_token) |value| secret.zeroAndFree(alloc, value);
    const token_url = try alloc.dupe(u8, input.token_url);
    errdefer alloc.free(token_url);
    const client_id = try alloc.dupe(u8, input.client_id);
    errdefer alloc.free(client_id);
    const scope: ?[]u8 = if (input.scope) |value| try alloc.dupe(u8, value) else null;
    errdefer if (scope) |value| alloc.free(value);
    const account_id: ?[]u8 = if (input.account_id) |value| try alloc.dupe(u8, value) else null;
    errdefer if (account_id) |value| alloc.free(value);

    return .{
        .provider = provider,
        .access_token = access_token,
        .refresh_token = refresh_token,
        .expires_at_ms = input.expires_at_ms,
        .token_url = token_url,
        .client_id = client_id,
        .scope = scope,
        .account_id = account_id,
    };
}

fn dupeRequiredString(alloc: Allocator, object: std.json.ObjectMap, key: []const u8) ![]u8 {
    const value = object.get(key) orelse return FlowError.InvalidTokenResponse;
    if (value != .string or value.string.len == 0) return FlowError.InvalidTokenResponse;
    return alloc.dupe(u8, value.string);
}

fn dupeOptionalString(alloc: Allocator, object: std.json.ObjectMap, key: []const u8) !?[]u8 {
    const value = object.get(key) orelse return null;
    if (value == .null) return null;
    if (value != .string or value.string.len == 0) return FlowError.InvalidTokenResponse;
    return try alloc.dupe(u8, value.string);
}

// --- Tests --------------------------------------------------------------

const testing = std.testing;

const Step = struct {
    method: ?oauth_transport.Method = null,
    url: ?[]const u8 = null,
    payload: ?[]const u8 = null,
    disposition: oauth_transport.Disposition = .accepted,
    body: []const u8 = "{}",
};

const ScriptedTransport = struct {
    steps: []const Step,
    index: usize = 0,
    mismatch: ?usize = null,

    fn provider(self: *ScriptedTransport) oauth_transport.Provider {
        return .{ .context = self, .execute_fn = execute };
    }

    fn execute(
        raw: ?*anyopaque,
        alloc: Allocator,
        request: oauth_transport.Request,
    ) anyerror!oauth_transport.Response {
        const self: *ScriptedTransport = @ptrCast(@alignCast(raw.?));
        if (self.index >= self.steps.len) return error.TestUnexpectedRequest;
        const step = self.steps[self.index];
        if (step.method) |method| {
            if (method != request.method) self.mismatch = self.index;
        }
        if (step.url) |url| {
            if (!std.mem.eql(u8, url, request.url)) self.mismatch = self.index;
        }
        if (step.payload) |payload| {
            const actual = request.payload orelse "";
            if (!std.mem.eql(u8, payload, actual)) self.mismatch = self.index;
        }
        self.index += 1;
        return .{
            .disposition = step.disposition,
            .body = try alloc.dupe(u8, step.body),
        };
    }
};

const RejectingTransport = struct {
    calls: usize = 0,

    fn provider(self: *RejectingTransport) oauth_transport.Provider {
        return .{ .context = self, .execute_fn = execute };
    }

    fn execute(
        raw: ?*anyopaque,
        _: Allocator,
        _: oauth_transport.Request,
    ) anyerror!oauth_transport.Response {
        const self: *RejectingTransport = @ptrCast(@alignCast(raw.?));
        self.calls += 1;
        return error.TestUnexpectedRequest;
    }
};

/// One context for every injected dependency: a fake clock that records the
/// sleeps, a silent terminal, no browser, and a session store pointed at a
/// temp directory through `auth_store`'s directory seam.
const TestEnv = struct {
    now: i64 = 0,
    sleeps: [16]u64 = @splat(0),
    sleep_count: usize = 0,
    saves: usize = 0,
    dir: ?*io_mod.VerifiedDir = null,
    messages: std.ArrayList(u8) = .empty,

    fn deinit(self: *TestEnv) void {
        self.messages.deinit(testing.allocator);
    }

    fn deps(self: *TestEnv) Deps {
        return .{
            .ctx = self,
            .now_ms = nowMs,
            .sleep_ms = sleepMs,
            .notify_fn = notify,
            .open_url_fn = openUrl,
            .store = .{ .ctx = self, .save_fn = save },
        };
    }

    fn state(raw: ?*anyopaque) *TestEnv {
        return @ptrCast(@alignCast(raw.?));
    }

    fn nowMs(raw: ?*anyopaque) i64 {
        return state(raw).now;
    }

    fn sleepMs(raw: ?*anyopaque, millis: u64) void {
        const self = state(raw);
        if (self.sleep_count < self.sleeps.len) {
            self.sleeps[self.sleep_count] = millis;
            self.sleep_count += 1;
        }
        self.now +|= @intCast(millis);
    }

    fn notify(raw: ?*anyopaque, text: []const u8) void {
        state(raw).messages.appendSlice(testing.allocator, text) catch {};
    }

    fn openUrl(_: ?*anyopaque, _: Allocator, _: []const u8) bool {
        return false;
    }

    fn save(raw: ?*anyopaque, alloc: Allocator, session: auth_store.Session) anyerror!void {
        const self = state(raw);
        self.saves += 1;
        const dir = self.dir orelse return;
        return auth_store.saveToDir(alloc, dir, session);
    }
};

fn openTestProfileDir(tmp: *std.testing.TmpDir) !io_mod.VerifiedDir {
    return .{ .dir = try tmp.dir.openDir(testing.io, ".", .{ .iterate = true }) };
}

/// A JWT with an unsigned, syntactically valid payload. Only the payload
/// segment is ever read, so the header and signature are placeholders.
fn craftJwt(alloc: Allocator, payload_json: []const u8) ![]u8 {
    const encoded_len = b64.Encoder.calcSize(payload_json.len);
    const buffer = try alloc.alloc(u8, encoded_len);
    defer alloc.free(buffer);
    const encoded = b64.Encoder.encode(buffer, payload_json);
    return std.fmt.allocPrint(alloc, "header.{s}.signature", .{encoded});
}

test "pkce challenge matches the RFC 7636 appendix B vector" {
    const verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
    const challenge = pkceChallenge(verifier);
    try testing.expectEqualStrings("E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM", &challenge);
}

test "generated pkce verifiers are base64url and hash to their challenge" {
    var first = generatePkce();
    defer first.zero();
    var second = generatePkce();
    defer second.zero();

    try testing.expect(!std.mem.eql(u8, &first.verifier, &second.verifier));
    for (first.verifier) |byte| {
        const safe = std.ascii.isAlphanumeric(byte) or byte == '-' or byte == '_';
        try testing.expect(safe);
    }
    try testing.expectEqualStrings(&pkceChallenge(&first.verifier), &first.challenge);

    const state = randomStateHex();
    try testing.expectEqual(@as(usize, 32), state.len);
    for (state) |byte| try testing.expect(std.ascii.isHex(byte));
}

test "account id is read from the id token, then the access token" {
    const alloc = testing.allocator;

    const with_account = try craftJwt(
        alloc,
        "{\"sub\":\"user\",\"" ++ openai_auth_claim ++ "\":{\"chatgpt_account_id\":\"acct_from_id\"}}",
    );
    defer alloc.free(with_account);
    const other_account = try craftJwt(
        alloc,
        "{\"" ++ openai_auth_claim ++ "\":{\"chatgpt_account_id\":\"acct_from_access\"}}",
    );
    defer alloc.free(other_account);
    const without_claim = try craftJwt(alloc, "{\"sub\":\"user\"}");
    defer alloc.free(without_claim);

    const from_id = accountIdFromJwt(alloc, with_account).?;
    defer alloc.free(from_id);
    try testing.expectEqualStrings("acct_from_id", from_id);

    var prefers_id = Tokens{
        .access_token = try alloc.dupe(u8, other_account),
        .id_token = try alloc.dupe(u8, with_account),
        .expires_in = 60,
    };
    defer prefers_id.deinit(alloc);
    const preferred = accountIdFromTokens(alloc, prefers_id).?;
    defer alloc.free(preferred);
    try testing.expectEqualStrings("acct_from_id", preferred);

    var falls_back = Tokens{
        .access_token = try alloc.dupe(u8, other_account),
        .id_token = try alloc.dupe(u8, without_claim),
        .expires_in = 60,
    };
    defer falls_back.deinit(alloc);
    const fallback = accountIdFromTokens(alloc, falls_back).?;
    defer alloc.free(fallback);
    try testing.expectEqualStrings("acct_from_access", fallback);

    try testing.expect(accountIdFromJwt(alloc, without_claim) == null);
    try testing.expect(accountIdFromJwt(alloc, "not-a-jwt") == null);
    try testing.expect(accountIdFromJwt(alloc, "header..signature") == null);
    try testing.expect(accountIdFromJwt(alloc, "header.!!!.signature") == null);
}

test "device login polls through pending and slow_down, then stores the session" {
    const alloc = testing.allocator;
    const def = registry.byKey("xai").?;

    var steps = [_]Step{
        .{
            .method = .get,
            .url = "https://auth.x.ai/.well-known/openid-configuration",
            .body = "{\"issuer\":\"https://auth.x.ai\"," ++
                "\"device_authorization_endpoint\":\"https://auth.x.ai/oauth2/device/code\"," ++
                "\"token_endpoint\":\"https://auth.x.ai/oauth2/token\"}",
        },
        .{
            .method = .post_form,
            .url = "https://auth.x.ai/oauth2/device/code",
            .payload = "client_id=b1a00492-073a-47ea-816f-4c329264a828&scope=openid%20profile%20" ++
                "email%20offline_access%20grok-cli%3Aaccess%20api%3Aaccess",
            .body = "{\"device_code\":\"device-code\",\"user_code\":\"ABCD-EFGH\"," ++
                "\"verification_uri\":\"https://x.ai/device\"," ++
                "\"verification_uri_complete\":\"https://x.ai/device?code=ABCD-EFGH\"," ++
                "\"expires_in\":600,\"interval\":1}",
        },
        .{
            .method = .post_form,
            .url = "https://auth.x.ai/oauth2/token",
            .disposition = .rejected,
            .body = "{\"error\":\"authorization_pending\"}",
        },
        .{
            .method = .post_form,
            .disposition = .rejected,
            .body = "{\"error\":\"slow_down\"}",
        },
        .{
            .method = .post_form,
            .body = "{\"access_token\":\"xai-access\",\"refresh_token\":\"xai-refresh\"," ++
                "\"expires_in\":3600,\"scope\":\"openid api:access\",\"token_type\":\"Bearer\"}",
        },
    };
    var transport = ScriptedTransport{ .steps = &steps };

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir = try openTestProfileDir(&tmp);
    defer dir.close();

    var env = TestEnv{ .dir = &dir };
    defer env.deinit();

    const outcome = try runLoginWith(alloc, transport.provider(), def, env.deps());
    try testing.expect(transport.mismatch == null);
    try testing.expectEqual(@as(usize, steps.len), transport.index);
    try testing.expectEqualStrings("xAI", outcome.provider_label);
    try testing.expect(!outcome.account_id_present);

    // One second between the first two polls, then five more after slow_down.
    try testing.expectEqual(@as(usize, 3), env.sleep_count);
    try testing.expectEqual(@as(u64, 1000), env.sleeps[0]);
    try testing.expectEqual(@as(u64, 1000), env.sleeps[1]);
    try testing.expectEqual(@as(u64, 6000), env.sleeps[2]);

    try testing.expect(std.mem.find(u8, env.messages.items, "ABCD-EFGH") != null);
    try testing.expect(std.mem.find(u8, env.messages.items, "https://x.ai/device?code=") != null);

    try testing.expectEqual(@as(usize, 1), env.saves);
    var stored = (try auth_store.loadFromDir(alloc, &dir.dir, "xai")).?;
    defer stored.deinit(alloc);
    try testing.expectEqualStrings("xai", stored.provider);
    try testing.expectEqualStrings("xai-access", stored.access_token);
    try testing.expectEqualStrings("xai-refresh", stored.refresh_token.?);
    try testing.expectEqualStrings("https://auth.x.ai/oauth2/token", stored.token_url);
    try testing.expectEqualStrings(def.oauth.client_id, stored.client_id);
    try testing.expectEqualStrings("openid api:access", stored.scope.?);
    try testing.expect(stored.account_id == null);
    // 8 seconds of fake sleeps elapsed before the grant landed.
    try testing.expectEqual(@as(i64, 8000 + 3600 * 1000), stored.expires_at_ms);
}

test "device login refuses a discovery document that leaves the issuer domain" {
    const alloc = testing.allocator;
    const def = registry.byKey("xai").?;

    var steps = [_]Step{.{
        .body = "{\"issuer\":\"https://auth.x.ai\"," ++
            "\"device_authorization_endpoint\":\"https://auth.x.ai.evil.example/device\"," ++
            "\"token_endpoint\":\"https://auth.x.ai/oauth2/token\"}",
    }};
    var transport = ScriptedTransport{ .steps = &steps };
    var env = TestEnv{};
    defer env.deinit();

    try testing.expectError(
        FlowError.UntrustedOAuthEndpoint,
        runLoginWith(alloc, transport.provider(), def, env.deps()),
    );
    try testing.expectEqual(@as(usize, 0), env.saves);
}

test "issuer endpoint trust follows the registrable domain and https" {
    try validateIssuerEndpoint("https://auth.x.ai", "https://auth.x.ai/oauth2/token");
    try validateIssuerEndpoint("https://auth.x.ai", "https://x.ai/oauth2/token");
    try validateIssuerEndpoint("https://auth.x.ai", "https://api.auth.x.ai/token");
    try validateIssuerEndpoint("https://auth.openai.com", "https://auth.openai.com/oauth/token");

    const rejected = [_][]const u8{
        "http://auth.x.ai/oauth2/token",
        "https://auth.x.ai.evil.example/token",
        "https://evil.example/token",
        "https://xx.ai/token",
        "https://notx.ai/token",
        "https://user:pass@auth.x.ai/token",
        "ftp://auth.x.ai/token",
        "not a url",
    };
    for (rejected) |endpoint| {
        try testing.expectError(
            FlowError.UntrustedOAuthEndpoint,
            validateIssuerEndpoint("https://auth.x.ai", endpoint),
        );
    }

    try testing.expectEqualStrings("x.ai", registrableDomain("auth.x.ai").?);
    try testing.expectEqualStrings("x.ai", registrableDomain("x.ai").?);
    try testing.expectEqualStrings("openai.com", registrableDomain("auth.api.openai.com").?);
    try testing.expect(registrableDomain("localhost") == null);
    try testing.expect(registrableDomain("") == null);
    try testing.expect(registrableDomain("trailing.") == null);
}

test "authorize url carries the pkce parameters the codex flow expects" {
    const alloc = testing.allocator;
    const def = registry.byKey("openai").?;
    const url = try buildAuthorizeUrl(
        alloc,
        def,
        "http://localhost:1455/auth/callback",
        "challenge-value",
        "state-value",
    );
    defer alloc.free(url);

    try testing.expectEqualStrings(
        "https://auth.openai.com/oauth/authorize?response_type=code" ++
            "&client_id=app_EMoamEEZ73f0CkXaXp7hrann" ++
            "&redirect_uri=http%3A%2F%2Flocalhost%3A1455%2Fauth%2Fcallback" ++
            "&scope=openid%20profile%20email%20offline_access" ++
            "&code_challenge=challenge-value&code_challenge_method=S256" ++
            "&state=state-value&id_token_add_organizations=false" ++
            "&codex_cli_simplified_flow=true",
        url,
    );
}

test "callback request line parsing extracts code, state, and errors" {
    const alloc = testing.allocator;

    var granted = try parseCallbackRequestLine(
        alloc,
        "GET /auth/callback?code=abc%2F123&state=xyz&other=1 HTTP/1.1",
    );
    defer granted.deinit(alloc);
    try testing.expectEqualStrings("abc/123", granted.code.?);
    try testing.expectEqualStrings("xyz", granted.state.?);
    try testing.expect(granted.err_code == null);

    var denied = try parseCallbackRequestLine(
        alloc,
        "GET /auth/callback?error=access_denied&error_description=User+said+no HTTP/1.1",
    );
    defer denied.deinit(alloc);
    try testing.expectEqualStrings("access_denied", denied.err_code.?);
    try testing.expectEqualStrings("User said no", denied.err_description.?);
    try testing.expect(denied.code == null);

    var no_version = try parseCallbackRequestLine(alloc, "GET /auth/callback?code=abc");
    defer no_version.deinit(alloc);
    try testing.expectEqualStrings("abc", no_version.code.?);

    try testing.expectError(
        FlowError.InvalidCallbackRequest,
        parseCallbackRequestLine(alloc, "POST /auth/callback?code=abc HTTP/1.1"),
    );
    try testing.expectError(
        FlowError.InvalidCallbackRequest,
        parseCallbackRequestLine(alloc, "GET /auth/callback HTTP/1.1"),
    );
    try testing.expectError(
        FlowError.InvalidCallbackRequest,
        parseCallbackRequestLine(alloc, "GET /auth/callback?code=%zz HTTP/1.1"),
    );
}

const CallbackClient = struct {
    io_backend: std.Io.Threaded = .init_single_threaded,
    port: u16,
    request: []const u8,
    failure: ?anyerror = null,

    fn run(self: *CallbackClient) void {
        self.runFallible() catch |err| {
            self.failure = err;
        };
    }

    fn runFallible(self: *CallbackClient) !void {
        const zio = self.io_backend.io();
        const address = std.Io.net.IpAddress{ .ip4 = .loopback(self.port) };
        var stream = try address.connect(zio, .{ .mode = .stream });
        defer stream.close(zio);

        var write_buffer: [512]u8 = undefined;
        var writer = stream.writer(zio, &write_buffer);
        try writer.interface.writeAll(self.request);
        try writer.interface.flush();

        var read_buffer: [1024]u8 = undefined;
        var reader = stream.reader(zio, &read_buffer);
        const response = reader.interface.allocRemaining(testing.allocator, .limited(4096)) catch return;
        testing.allocator.free(response);
    }
};

test "loopback listener accepts one callback and parses it" {
    const alloc = testing.allocator;
    var listener = try Loopback.bind(0);
    defer listener.deinit();

    var client = CallbackClient{
        .port = listener.port(),
        .request = "GET /auth/callback?code=loopback-code&state=loopback-state HTTP/1.1\r\n" ++
            "Host: localhost\r\nConnection: close\r\n\r\n",
    };
    const thread = try std.Thread.spawn(.{}, CallbackClient.run, .{&client});
    defer thread.join();

    var callback = try listener.awaitCallback(alloc, 10_000);
    defer callback.deinit(alloc);
    try testing.expectEqualStrings("loopback-code", callback.code.?);
    try testing.expectEqualStrings("loopback-state", callback.state.?);
}

test "loopback listener times out without a callback" {
    const alloc = testing.allocator;
    var listener = try Loopback.bind(0);
    defer listener.deinit();
    try testing.expect(listener.port() != 0);
    try testing.expectError(FlowError.CallbackTimedOut, listener.awaitCallback(alloc, 1));
}

test "refresh is skipped while the session is outside the skew window" {
    const alloc = testing.allocator;
    const def = registry.byKey("openai").?;
    var transport = RejectingTransport{};

    var session = auth_store.Session{
        .provider = try alloc.dupe(u8, "openai"),
        .access_token = try alloc.dupe(u8, "access"),
        .refresh_token = try alloc.dupe(u8, "refresh"),
        .expires_at_ms = 500_000,
        .token_url = try alloc.dupe(u8, def.oauth.token_url),
        .client_id = try alloc.dupe(u8, def.oauth.client_id),
    };
    defer session.deinit(alloc);

    var env = TestEnv{ .now = 439_999 };
    defer env.deinit();
    try testing.expect(!try refreshIfNeededWith(
        alloc,
        transport.provider(),
        def,
        &session,
        env.deps(),
    ));
    try testing.expectEqual(@as(usize, 0), transport.calls);
    try testing.expectEqual(@as(usize, 0), env.saves);
}

test "refresh exchanges inside the skew window and persists a rotated token" {
    const alloc = testing.allocator;
    const def = registry.byKey("xai").?;

    const id_token = try craftJwt(
        alloc,
        "{\"" ++ openai_auth_claim ++ "\":{\"chatgpt_account_id\":\"acct_new\"}}",
    );
    defer alloc.free(id_token);
    const body = try std.fmt.allocPrint(
        alloc,
        "{{\"access_token\":\"fresh-access\",\"refresh_token\":\"rotated-refresh\"," ++
            "\"id_token\":\"{s}\",\"expires_in\":1800,\"token_type\":\"Bearer\"}}",
        .{id_token},
    );
    defer alloc.free(body);

    var steps = [_]Step{.{
        .method = .post_form,
        .url = "https://auth.x.ai/oauth2/token",
        .payload = "grant_type=refresh_token&refresh_token=old-refresh" ++
            "&client_id=b1a00492-073a-47ea-816f-4c329264a828",
        .body = body,
    }};
    var transport = ScriptedTransport{ .steps = &steps };

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir = try openTestProfileDir(&tmp);
    defer dir.close();

    var session = auth_store.Session{
        .provider = try alloc.dupe(u8, "xai"),
        .access_token = try alloc.dupe(u8, "stale-access"),
        .refresh_token = try alloc.dupe(u8, "old-refresh"),
        .expires_at_ms = 100_000,
        .token_url = try alloc.dupe(u8, "https://auth.x.ai/oauth2/token"),
        .client_id = try alloc.dupe(u8, def.oauth.client_id),
    };
    defer session.deinit(alloc);

    var env = TestEnv{ .now = 90_000, .dir = &dir };
    defer env.deinit();

    try testing.expect(try refreshIfNeededWith(
        alloc,
        transport.provider(),
        def,
        &session,
        env.deps(),
    ));
    try testing.expect(transport.mismatch == null);
    try testing.expectEqualStrings("fresh-access", session.access_token);
    try testing.expectEqualStrings("rotated-refresh", session.refresh_token.?);
    try testing.expectEqualStrings("acct_new", session.account_id.?);
    try testing.expectEqual(@as(i64, 90_000 + 1_800_000), session.expires_at_ms);

    var stored = (try auth_store.loadFromDir(alloc, &dir.dir, "xai")).?;
    defer stored.deinit(alloc);
    try testing.expectEqualStrings("fresh-access", stored.access_token);
    try testing.expectEqualStrings("rotated-refresh", stored.refresh_token.?);
}

test "openai refresh narrows the scope and keeps the existing refresh token" {
    const alloc = testing.allocator;
    const def = registry.byKey("openai").?;

    var steps = [_]Step{.{
        .method = .post_form,
        .url = "https://auth.openai.com/oauth/token",
        .payload = "grant_type=refresh_token&refresh_token=keep-me" ++
            "&client_id=app_EMoamEEZ73f0CkXaXp7hrann&scope=openid%20profile%20email",
        .body = "{\"access_token\":\"fresh\",\"expires_in\":3600,\"token_type\":\"Bearer\"}",
    }};
    var transport = ScriptedTransport{ .steps = &steps };

    var session = auth_store.Session{
        .provider = try alloc.dupe(u8, "openai"),
        .access_token = try alloc.dupe(u8, "stale"),
        .refresh_token = try alloc.dupe(u8, "keep-me"),
        .expires_at_ms = 0,
        .token_url = try alloc.dupe(u8, def.oauth.token_url),
        .client_id = try alloc.dupe(u8, def.oauth.client_id),
        .account_id = try alloc.dupe(u8, "acct_existing"),
    };
    defer session.deinit(alloc);

    var env = TestEnv{ .now = 1_000 };
    defer env.deinit();
    try testing.expect(try refreshIfNeededWith(
        alloc,
        transport.provider(),
        def,
        &session,
        env.deps(),
    ));
    try testing.expect(transport.mismatch == null);
    try testing.expectEqualStrings("fresh", session.access_token);
    try testing.expectEqualStrings("keep-me", session.refresh_token.?);
    try testing.expectEqualStrings("acct_existing", session.account_id.?);
    try testing.expectEqual(@as(usize, 1), env.saves);
}

test "a rejected refresh token asks for a new sign-in" {
    const alloc = testing.allocator;
    const def = registry.byKey("openai").?;

    var steps = [_]Step{.{
        .disposition = .rejected,
        .body = "{\"error\":\"invalid_grant\",\"error_description\":\"refresh token expired\"}",
    }};
    var transport = ScriptedTransport{ .steps = &steps };

    var session = auth_store.Session{
        .provider = try alloc.dupe(u8, "openai"),
        .access_token = try alloc.dupe(u8, "stale"),
        .refresh_token = try alloc.dupe(u8, "dead"),
        .expires_at_ms = 0,
        .token_url = try alloc.dupe(u8, def.oauth.token_url),
        .client_id = try alloc.dupe(u8, def.oauth.client_id),
    };
    defer session.deinit(alloc);

    var env = TestEnv{ .now = 1_000 };
    defer env.deinit();
    try testing.expectError(FlowError.ReauthenticationRequired, refreshIfNeededWith(
        alloc,
        transport.provider(),
        def,
        &session,
        env.deps(),
    ));
    try testing.expectEqualStrings("stale", session.access_token);
    try testing.expectEqual(@as(usize, 0), env.saves);
}

test "a session without a refresh token asks for a new sign-in" {
    const alloc = testing.allocator;
    const def = registry.byKey("openai").?;
    var transport = RejectingTransport{};

    var session = auth_store.Session{
        .provider = try alloc.dupe(u8, "openai"),
        .access_token = try alloc.dupe(u8, "stale"),
        .expires_at_ms = 0,
        .token_url = try alloc.dupe(u8, def.oauth.token_url),
        .client_id = try alloc.dupe(u8, def.oauth.client_id),
    };
    defer session.deinit(alloc);

    var env = TestEnv{ .now = 1_000 };
    defer env.deinit();
    try testing.expectError(FlowError.ReauthenticationRequired, refreshIfNeededWith(
        alloc,
        transport.provider(),
        def,
        &session,
        env.deps(),
    ));
    try testing.expectEqual(@as(usize, 0), transport.calls);
}

test "token responses tolerate a missing expires_in and reject a wrong token type" {
    const alloc = testing.allocator;

    var lenient = try parseTokens(alloc, "{\"access_token\":\"a\"}");
    defer lenient.deinit(alloc);
    try testing.expectEqual(default_expires_in_s, lenient.expires_in);
    try testing.expect(lenient.refresh_token == null);

    try testing.expectError(
        FlowError.InvalidTokenResponse,
        parseTokens(alloc, "{\"access_token\":\"a\",\"token_type\":\"mac\"}"),
    );
    try testing.expectError(
        FlowError.InvalidTokenResponse,
        parseTokens(alloc, "{\"access_token\":\"a\",\"expires_in\":0}"),
    );
    try testing.expectError(FlowError.InvalidTokenResponse, parseTokens(alloc, "{}"));
    try testing.expectError(FlowError.InvalidTokenResponse, parseTokens(alloc, "[]"));
}

test "oauth error bodies map to actionable errors" {
    const alloc = testing.allocator;
    try testing.expectEqual(
        oauth.OAuthError.AuthorizationPending,
        mapErrorBody(alloc, "{\"error\":\"authorization_pending\"}"),
    );
    try testing.expectEqual(
        oauth.OAuthError.SlowDown,
        mapErrorBody(alloc, "{\"error\":\"slow_down\"}"),
    );
    try testing.expectEqual(
        oauth.OAuthError.AccessDenied,
        mapErrorBody(alloc, "{\"error\":\"access_denied\"}"),
    );
    try testing.expectEqual(
        FlowError.ReauthenticationRequired,
        mapErrorBody(alloc, "{\"error\":\"invalid_grant\"}"),
    );
    try testing.expectEqual(
        oauth.OAuthError.OAuthRequestFailed,
        mapErrorBody(alloc, "not json"),
    );
}
