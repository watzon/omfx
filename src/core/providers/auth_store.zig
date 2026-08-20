//! omfx fork-owned OAuth session storage for direct providers.
//!
//! Upstream keeps the single Vercel session in `~/.fx/auth.json`
//! (`core/auth/oauth_session.zig`). The fork adds one file per direct
//! provider at `~/.fx/auth-<provider>.json`, guarded by
//! `~/.fx/auth-<provider>.lock`, so signing in to OpenAI never disturbs the
//! Vercel session or an xAI session.
//!
//! The durability rules mirror the upstream store exactly: 0o600 files, a
//! group/world-readable file is refused instead of trusted, reads are bounded
//! at 64 KiB, mutations take a short advisory lock, and writes land through an
//! atomic durable replace. Direct providers are native-only, so on wasm hosts
//! `load` reports "no session" and `save` fails.

const std = @import("std");
const secret = @import("../auth/secret.zig");
const host_target = @import("../hosts/target.zig");
const debug_trace = @import("../shared/debug_trace.zig");
const io_mod = @import("../shared/io.zig");
const profile_paths = @import("../shared/profile_paths.zig");

const Allocator = std.mem.Allocator;

pub const schema_version: i64 = 1;

const max_auth_file_bytes: usize = 64 * 1024;
const expiry_skew_ms: i64 = 60 * std.time.ms_per_s;
const mutation_lock_deadline_ms: u64 = 2000;
const max_provider_key_len: usize = 32;
const file_name_prefix = "auth-";
const file_name_suffix = ".json";
const lock_name_suffix = ".lock";
const name_buf_len = file_name_prefix.len + max_provider_key_len + file_name_suffix.len;

/// A stored OAuth session for one direct provider.
///
/// Every slice is owned by the session. `deinit` wipes the token bytes before
/// releasing them.
pub const Session = struct {
    /// Registry definition key, for example "openai" or "xai".
    provider: []u8,
    access_token: []u8,
    refresh_token: ?[]u8 = null,
    expires_at_ms: i64,
    /// Endpoint used to refresh this session.
    token_url: []u8,
    client_id: []u8,
    scope: ?[]u8 = null,
    /// ChatGPT account id for openai. Null for providers that do not use one.
    account_id: ?[]u8 = null,

    pub fn deinit(self: *Session, alloc: Allocator) void {
        alloc.free(self.provider);
        secret.zeroAndFree(alloc, self.access_token);
        if (self.refresh_token) |value| secret.zeroAndFree(alloc, value);
        alloc.free(self.token_url);
        alloc.free(self.client_id);
        if (self.scope) |value| alloc.free(value);
        if (self.account_id) |value| alloc.free(value);
        self.* = undefined;
    }

    /// True when the access token is gone or close enough to expiry that the
    /// next request would race the skew window.
    pub fn expired(self: Session, now_ms: i64) bool {
        return refreshDeadlineMs(self.expires_at_ms) <= now_ms;
    }
};

/// The instant a caller must refresh by: expiry minus a 60 second skew.
pub fn refreshDeadlineMs(expires_at_ms: i64) i64 {
    return expires_at_ms -| expiry_skew_ms;
}

/// Reads the stored session for `provider_key`, or null when there is none.
///
/// A malformed file, a wrong schema version, or insecure permissions all log
/// and report "no session" rather than failing the caller.
pub fn load(alloc: Allocator, provider_key: []const u8) !?Session {
    try validateProviderKey(provider_key);
    if (comptime host_target.is_wasm) {
        debug_trace.logf(
            "auth",
            "provider session load skipped provider={s} step=host err=DirectProviderAuthUnsupported",
            .{provider_key},
        );
        return null;
    }
    var fx_dir = openProfileDirForRead() orelse return null;
    defer fx_dir.close(io_mod.getIo());
    return loadFromDir(alloc, &fx_dir, provider_key);
}

/// Writes `session` durably, creating `~/.fx` with 0o700 when it is missing.
pub fn save(alloc: Allocator, session: Session) !void {
    try validateProviderKey(session.provider);
    if (comptime host_target.is_wasm) return error.DirectProviderAuthUnsupported;
    var fx_dir = try openProfileDirForWrite();
    defer fx_dir.close();
    return saveToDir(alloc, &fx_dir, session);
}

/// Removes the stored session. Returns true when a file was removed.
pub fn delete(provider_key: []const u8) !bool {
    try validateProviderKey(provider_key);
    if (comptime host_target.is_wasm) return error.DirectProviderAuthUnsupported;
    var fx_dir = openProfileDirForWriteExisting() catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    } orelse return false;
    defer fx_dir.close();
    return deleteFromDir(&fx_dir, provider_key);
}

/// Cheap presence check. Does not read or parse the file.
pub fn exists(provider_key: []const u8) bool {
    validateProviderKey(provider_key) catch return false;
    if (comptime host_target.is_wasm) return false;
    var fx_dir = openProfileDirForRead() orelse return false;
    defer fx_dir.close(io_mod.getIo());
    return existsInDir(&fx_dir, provider_key);
}

// --- Directory-injected seams -------------------------------------------
//
// The HOME-based wrappers above resolve `~/.fx` and then delegate here. Tests
// pass a `std.testing.tmpDir` handle instead, exactly like
// `oauth_session.loadFromDir` does upstream.

/// Reads the session for `provider_key` from an already open profile dir.
pub fn loadFromDir(alloc: Allocator, fx_dir: *std.Io.Dir, provider_key: []const u8) !?Session {
    try validateProviderKey(provider_key);
    var name_buf: [name_buf_len]u8 = undefined;
    const file_name = try sessionFileName(&name_buf, provider_key);
    const zio = io_mod.getIo();

    var file = fx_dir.openFile(zio, file_name, .{
        .mode = .read_only,
        .allow_directory = false,
        .follow_symlinks = false,
        .resolve_beneath = true,
    }) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => {
            logFailure(provider_key, "open_file", err);
            return null;
        },
    };
    defer file.close(zio);

    const stat = file.stat(zio) catch |err| {
        logFailure(provider_key, "stat", err);
        return null;
    };
    if (stat.kind != .file or stat.permissions.toMode() & 0o077 != 0) {
        debug_trace.logf(
            "auth",
            "provider session load failed provider={s} step=permissions err=InsecureAuthFile",
            .{provider_key},
        );
        return null;
    }

    const bytes = io_mod.readFileToEnd(alloc, &file, max_auth_file_bytes) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => {
            logFailure(provider_key, "read", err);
            return null;
        },
    };
    defer secret.zeroAndFree(alloc, bytes);

    return parse(alloc, bytes, provider_key) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => {
            logFailure(provider_key, "parse", err);
            return null;
        },
    };
}

/// Durably replaces the session file inside an already open profile dir.
pub fn saveToDir(alloc: Allocator, fx_dir: *io_mod.VerifiedDir, session: Session) !void {
    try validateProviderKey(session.provider);
    var name_buf: [name_buf_len]u8 = undefined;
    const file_name = try sessionFileName(&name_buf, session.provider);
    var lock_buf: [name_buf_len]u8 = undefined;
    const lock_name = try lockFileName(&lock_buf, session.provider);

    var lock = try io_mod.acquireTimedAdvisoryLock(fx_dir, lock_name, mutation_lock_deadline_ms);
    defer lock.release();

    const text = try stringify(alloc, session);
    defer secret.zeroAndFree(alloc, text);
    try io_mod.durableReplaceVerified(alloc, fx_dir, file_name, text);
}

/// Removes the session file inside an already open profile dir.
pub fn deleteFromDir(fx_dir: *io_mod.VerifiedDir, provider_key: []const u8) !bool {
    try validateProviderKey(provider_key);
    var name_buf: [name_buf_len]u8 = undefined;
    const file_name = try sessionFileName(&name_buf, provider_key);
    var lock_buf: [name_buf_len]u8 = undefined;
    const lock_name = try lockFileName(&lock_buf, provider_key);

    var lock = try io_mod.acquireTimedAdvisoryLock(fx_dir, lock_name, mutation_lock_deadline_ms);
    defer lock.release();

    fx_dir.dir.deleteFile(io_mod.getIo(), file_name) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    io_mod.syncVerifiedDir(fx_dir.dir) catch {};
    return true;
}

/// Presence check inside an already open profile dir.
pub fn existsInDir(fx_dir: *std.Io.Dir, provider_key: []const u8) bool {
    validateProviderKey(provider_key) catch return false;
    var name_buf: [name_buf_len]u8 = undefined;
    const file_name = sessionFileName(&name_buf, provider_key) catch return false;
    const stat = fx_dir.statFile(io_mod.getIo(), file_name, .{ .follow_symlinks = false }) catch return false;
    return stat.kind == .file;
}

// --- Naming and validation ----------------------------------------------

/// Provider keys become file names, so only `[a-z0-9_-]` is accepted.
fn validateProviderKey(provider_key: []const u8) !void {
    if (provider_key.len == 0 or provider_key.len > max_provider_key_len) {
        return error.InvalidProviderKey;
    }
    for (provider_key) |byte| {
        const allowed = (byte >= 'a' and byte <= 'z') or
            (byte >= '0' and byte <= '9') or
            byte == '_' or byte == '-';
        if (!allowed) return error.InvalidProviderKey;
    }
}

fn sessionFileName(buf: []u8, provider_key: []const u8) ![]const u8 {
    try validateProviderKey(provider_key);
    return std.fmt.bufPrint(buf, file_name_prefix ++ "{s}" ++ file_name_suffix, .{provider_key});
}

fn lockFileName(buf: []u8, provider_key: []const u8) ![]const u8 {
    try validateProviderKey(provider_key);
    return std.fmt.bufPrint(buf, file_name_prefix ++ "{s}" ++ lock_name_suffix, .{provider_key});
}

// --- Profile directory --------------------------------------------------

fn openProfileDirForRead() ?std.Io.Dir {
    const zio = io_mod.getIo();
    const home = io_mod.getenv("HOME") orelse {
        debug_trace.logf("auth", "provider session skipped step=home err=HomeNotSet", .{});
        return null;
    };
    var home_dir = std.Io.Dir.openDirAbsolute(zio, home, .{ .iterate = true }) catch |err| {
        debug_trace.logf("auth", "provider session failed step=open_home err={s}", .{@errorName(err)});
        return null;
    };
    defer home_dir.close(zio);

    return home_dir.openDir(zio, profile_paths.root_dir_name, .{
        .iterate = true,
        .follow_symlinks = false,
    }) catch |err| {
        debug_trace.logf("auth", "provider session failed step=open_profile err={s}", .{@errorName(err)});
        return null;
    };
}

fn openProfileDirForWrite() !io_mod.VerifiedDir {
    const home = io_mod.getenv("HOME") orelse return error.HomeNotSet;
    var home_dir = io_mod.VerifiedDir{
        .dir = try std.Io.Dir.openDirAbsolute(io_mod.getIo(), home, .{ .iterate = true }),
    };
    defer home_dir.close();
    return io_mod.openOrCreateVerifiedPrivateDir(&home_dir, profile_paths.root_dir_name);
}

fn openProfileDirForWriteExisting() !?io_mod.VerifiedDir {
    const zio = io_mod.getIo();
    const home = io_mod.getenv("HOME") orelse return error.HomeNotSet;
    var home_dir = std.Io.Dir.openDirAbsolute(zio, home, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer home_dir.close(zio);

    const dir = home_dir.openDir(zio, profile_paths.root_dir_name, .{
        .iterate = true,
        .follow_symlinks = false,
    }) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    return io_mod.VerifiedDir{ .dir = dir };
}

fn logFailure(provider_key: []const u8, step: []const u8, err: anyerror) void {
    debug_trace.logf(
        "auth",
        "provider session load failed provider={s} step={s} err={s}",
        .{ provider_key, step, @errorName(err) },
    );
}

// --- Serialization ------------------------------------------------------

fn stringify(alloc: Allocator, session: Session) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(alloc);
    errdefer out.deinit();
    const writer = &out.writer;
    try writer.print("{{\"version\":{d}", .{schema_version});
    try writeStringField(writer, "provider", session.provider);
    try writeStringField(writer, "access_token", session.access_token);
    try writeOptionalField(writer, "refresh_token", session.refresh_token);
    try writer.print(",\"expires_at_ms\":{d}", .{session.expires_at_ms});
    try writeStringField(writer, "token_url", session.token_url);
    try writeStringField(writer, "client_id", session.client_id);
    try writeOptionalField(writer, "scope", session.scope);
    try writeOptionalField(writer, "account_id", session.account_id);
    try writer.writeAll("}\n");
    return out.toOwnedSlice();
}

fn writeStringField(writer: *std.Io.Writer, name: []const u8, value: []const u8) !void {
    try writer.writeAll(",");
    try std.json.Stringify.value(name, .{}, writer);
    try writer.writeAll(":");
    try std.json.Stringify.value(value, .{}, writer);
}

fn writeOptionalField(writer: *std.Io.Writer, name: []const u8, value: ?[]const u8) !void {
    if (value) |present| return writeStringField(writer, name, present);
    try writer.writeAll(",");
    try std.json.Stringify.value(name, .{}, writer);
    try writer.writeAll(":null");
}

fn parse(alloc: Allocator, bytes: []const u8, expected_provider: []const u8) !Session {
    var parsed = try std.json.parseFromSlice(std.json.Value, alloc, bytes, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidProviderSession;
    const object = parsed.value.object;

    const version = object.get("version") orelse return error.InvalidProviderSession;
    if (version != .integer or version.integer != schema_version) {
        return error.InvalidProviderSession;
    }
    const stored_provider = try requiredString(object, "provider");
    if (!std.mem.eql(u8, stored_provider, expected_provider)) {
        return error.InvalidProviderSession;
    }
    const expires_at_ms = try requiredInteger(object, "expires_at_ms");

    const provider = try alloc.dupe(u8, stored_provider);
    errdefer alloc.free(provider);
    const access_token = try dupeRequiredString(alloc, object, "access_token");
    errdefer secret.zeroAndFree(alloc, access_token);
    const refresh_token = try dupeOptionalString(alloc, object, "refresh_token");
    errdefer if (refresh_token) |value| secret.zeroAndFree(alloc, value);
    const token_url = try dupeRequiredString(alloc, object, "token_url");
    errdefer alloc.free(token_url);
    const client_id = try dupeRequiredString(alloc, object, "client_id");
    errdefer alloc.free(client_id);
    const scope = try dupeOptionalString(alloc, object, "scope");
    errdefer if (scope) |value| alloc.free(value);
    const account_id = try dupeOptionalString(alloc, object, "account_id");
    errdefer if (account_id) |value| alloc.free(value);

    return .{
        .provider = provider,
        .access_token = access_token,
        .refresh_token = refresh_token,
        .expires_at_ms = expires_at_ms,
        .token_url = token_url,
        .client_id = client_id,
        .scope = scope,
        .account_id = account_id,
    };
}

fn requiredString(object: std.json.ObjectMap, key: []const u8) ![]const u8 {
    const value = object.get(key) orelse return error.InvalidProviderSession;
    if (value != .string or value.string.len == 0) return error.InvalidProviderSession;
    return value.string;
}

fn dupeRequiredString(alloc: Allocator, object: std.json.ObjectMap, key: []const u8) ![]u8 {
    return alloc.dupe(u8, try requiredString(object, key));
}

fn dupeOptionalString(alloc: Allocator, object: std.json.ObjectMap, key: []const u8) !?[]u8 {
    const value = object.get(key) orelse return null;
    if (value == .null) return null;
    if (value != .string or value.string.len == 0) return error.InvalidProviderSession;
    return try alloc.dupe(u8, value.string);
}

fn requiredInteger(object: std.json.ObjectMap, key: []const u8) !i64 {
    const value = object.get(key) orelse return error.InvalidProviderSession;
    if (value != .integer) return error.InvalidProviderSession;
    return value.integer;
}

// --- Tests --------------------------------------------------------------

const testing = std.testing;

fn testSession(alloc: Allocator) !Session {
    return .{
        .provider = try alloc.dupe(u8, "openai"),
        .access_token = try alloc.dupe(u8, "access-token"),
        .refresh_token = try alloc.dupe(u8, "refresh-token"),
        .expires_at_ms = 1_700_000_000_000,
        .token_url = try alloc.dupe(u8, "https://auth.openai.com/oauth/token"),
        .client_id = try alloc.dupe(u8, "app_client"),
        .scope = try alloc.dupe(u8, "openid profile email offline_access"),
        .account_id = try alloc.dupe(u8, "acct_123"),
    };
}

fn openTestProfileDir(tmp: *std.testing.TmpDir) !io_mod.VerifiedDir {
    return .{ .dir = try tmp.dir.openDir(testing.io, ".", .{ .iterate = true }) };
}

test "provider session round-trips through save, load, exists, and delete" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var fx_dir = try openTestProfileDir(&tmp);
    defer fx_dir.close();

    try testing.expect(!existsInDir(&fx_dir.dir, "openai"));
    try testing.expect((try loadFromDir(alloc, &fx_dir.dir, "openai")) == null);
    try testing.expect(!try deleteFromDir(&fx_dir, "openai"));

    var written = try testSession(alloc);
    defer written.deinit(alloc);
    try saveToDir(alloc, &fx_dir, written);

    try testing.expect(existsInDir(&fx_dir.dir, "openai"));
    var loaded = (try loadFromDir(alloc, &fx_dir.dir, "openai")).?;
    defer loaded.deinit(alloc);
    try testing.expectEqualStrings("openai", loaded.provider);
    try testing.expectEqualStrings("access-token", loaded.access_token);
    try testing.expectEqualStrings("refresh-token", loaded.refresh_token.?);
    try testing.expectEqual(@as(i64, 1_700_000_000_000), loaded.expires_at_ms);
    try testing.expectEqualStrings("https://auth.openai.com/oauth/token", loaded.token_url);
    try testing.expectEqualStrings("app_client", loaded.client_id);
    try testing.expectEqualStrings("openid profile email offline_access", loaded.scope.?);
    try testing.expectEqualStrings("acct_123", loaded.account_id.?);

    try testing.expect(try deleteFromDir(&fx_dir, "openai"));
    try testing.expect(!existsInDir(&fx_dir.dir, "openai"));
    try testing.expect(!try deleteFromDir(&fx_dir, "openai"));
}

test "provider sessions stay in separate files per provider" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var fx_dir = try openTestProfileDir(&tmp);
    defer fx_dir.close();

    var openai = try testSession(alloc);
    defer openai.deinit(alloc);
    try saveToDir(alloc, &fx_dir, openai);

    var xai = Session{
        .provider = try alloc.dupe(u8, "xai"),
        .access_token = try alloc.dupe(u8, "xai-access"),
        .expires_at_ms = 42,
        .token_url = try alloc.dupe(u8, "https://auth.x.ai/oauth2/token"),
        .client_id = try alloc.dupe(u8, "xai-client"),
    };
    defer xai.deinit(alloc);
    try saveToDir(alloc, &fx_dir, xai);

    try testing.expect(existsInDir(&fx_dir.dir, "openai"));
    try testing.expect(existsInDir(&fx_dir.dir, "xai"));

    var loaded = (try loadFromDir(alloc, &fx_dir.dir, "xai")).?;
    defer loaded.deinit(alloc);
    try testing.expectEqualStrings("xai-access", loaded.access_token);
    try testing.expect(loaded.refresh_token == null);
    try testing.expect(loaded.scope == null);
    try testing.expect(loaded.account_id == null);

    try testing.expect(try deleteFromDir(&fx_dir, "xai"));
    try testing.expect(existsInDir(&fx_dir.dir, "openai"));
}

test "provider session file is created group and world unreadable" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var fx_dir = try openTestProfileDir(&tmp);
    defer fx_dir.close();

    var session = try testSession(alloc);
    defer session.deinit(alloc);
    try saveToDir(alloc, &fx_dir, session);

    const stat = try fx_dir.dir.statFile(testing.io, "auth-openai.json", .{ .follow_symlinks = false });
    try testing.expectEqual(@as(u32, 0o600), stat.permissions.toMode() & 0o777);
}

test "provider session load refuses a group readable file" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var file = try tmp.dir.createFile(testing.io, "auth-openai.json", .{
        .permissions = std.Io.File.Permissions.fromMode(0o644),
    });
    try file.writeStreamingAll(
        testing.io,
        "{\"version\":1,\"provider\":\"openai\",\"access_token\":\"a\",\"expires_at_ms\":1,\"token_url\":\"https://t\",\"client_id\":\"c\"}",
    );
    file.close(testing.io);

    try testing.expect((try loadFromDir(alloc, &tmp.dir, "openai")) == null);
}

test "provider session load tolerates malformed and mismatched files" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const cases = [_][]const u8{
        "not json at all",
        "[]",
        "{\"version\":2,\"provider\":\"openai\",\"access_token\":\"a\",\"expires_at_ms\":1,\"token_url\":\"https://t\",\"client_id\":\"c\"}",
        "{\"version\":1,\"provider\":\"xai\",\"access_token\":\"a\",\"expires_at_ms\":1,\"token_url\":\"https://t\",\"client_id\":\"c\"}",
        "{\"version\":1,\"provider\":\"openai\",\"expires_at_ms\":1,\"token_url\":\"https://t\",\"client_id\":\"c\"}",
    };
    for (cases) |body| {
        var file = try tmp.dir.createFile(testing.io, "auth-openai.json", .{
            .truncate = true,
            .permissions = std.Io.File.Permissions.fromMode(0o600),
        });
        try file.writeStreamingAll(testing.io, body);
        file.close(testing.io);
        try testing.expect((try loadFromDir(alloc, &tmp.dir, "openai")) == null);
    }
}

test "provider keys that are not safe file names are refused" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const rejected = [_][]const u8{
        "",
        "../escape",
        "open/ai",
        "OpenAI",
        "open ai",
        "open.ai",
        "a" ** (max_provider_key_len + 1),
    };
    for (rejected) |key| {
        try testing.expectError(error.InvalidProviderKey, validateProviderKey(key));
        try testing.expectError(error.InvalidProviderKey, loadFromDir(alloc, &tmp.dir, key));
        try testing.expect(!existsInDir(&tmp.dir, key));
    }

    const accepted = [_][]const u8{ "openai", "xai", "z-ai", "vendor_9" };
    for (accepted) |key| try validateProviderKey(key);
}

test "provider session file names follow the fork layout" {
    var buf: [name_buf_len]u8 = undefined;
    try testing.expectEqualStrings("auth-openai.json", try sessionFileName(&buf, "openai"));
    try testing.expectEqualStrings("auth-xai.lock", try lockFileName(&buf, "xai"));
}

test "provider session serializes absent optionals as null" {
    const alloc = testing.allocator;
    var session = Session{
        .provider = try alloc.dupe(u8, "xai"),
        .access_token = try alloc.dupe(u8, "access"),
        .expires_at_ms = 7,
        .token_url = try alloc.dupe(u8, "https://auth.x.ai/oauth2/token"),
        .client_id = try alloc.dupe(u8, "client"),
    };
    defer session.deinit(alloc);

    const text = try stringify(alloc, session);
    defer secret.zeroAndFree(alloc, text);
    try testing.expectEqualStrings(
        "{\"version\":1,\"provider\":\"xai\",\"access_token\":\"access\",\"refresh_token\":null," ++
            "\"expires_at_ms\":7,\"token_url\":\"https://auth.x.ai/oauth2/token\"," ++
            "\"client_id\":\"client\",\"scope\":null,\"account_id\":null}\n",
        text,
    );

    var parsed = try parse(alloc, text, "xai");
    defer parsed.deinit(alloc);
    try testing.expect(parsed.refresh_token == null);
    try testing.expect(parsed.scope == null);
    try testing.expect(parsed.account_id == null);
}

test "provider session refresh deadline subtracts the expiry skew" {
    try testing.expectEqual(@as(i64, 40_000), refreshDeadlineMs(100_000));
    try testing.expectEqual(std.math.minInt(i64), refreshDeadlineMs(std.math.minInt(i64)));

    var session = Session{
        .provider = @constCast("openai"),
        .access_token = @constCast("access"),
        .expires_at_ms = 100_000,
        .token_url = @constCast("https://auth.openai.com/oauth/token"),
        .client_id = @constCast("client"),
    };
    try testing.expect(session.expired(50_000));
    try testing.expect(!session.expired(1));
    try testing.expect(session.expired(std.math.maxInt(i64)));
}

fn checkParseAllocationFailures(alloc: Allocator) !void {
    var session = try parse(
        alloc,
        "{\"version\":1,\"provider\":\"openai\",\"access_token\":\"a\",\"refresh_token\":\"r\"," ++
            "\"expires_at_ms\":1,\"token_url\":\"https://t\",\"client_id\":\"c\"," ++
            "\"scope\":\"openid\",\"account_id\":\"acct\"}",
        "openai",
    );
    defer session.deinit(alloc);
}

test "provider session parse cleans up allocation failures" {
    try testing.checkAllAllocationFailures(testing.allocator, checkParseAllocationFailures, .{});
}

test "provider session mutation lock serializes independent handles" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var first = try openTestProfileDir(&tmp);
    defer first.close();
    var second = try openTestProfileDir(&tmp);
    defer second.close();

    var held = try io_mod.acquireTimedAdvisoryLock(&first, "auth-openai.lock", 0);
    defer held.release();
    try testing.expectError(
        error.LockBusy,
        io_mod.acquireTimedAdvisoryLock(&second, "auth-openai.lock", 0),
    );
}
