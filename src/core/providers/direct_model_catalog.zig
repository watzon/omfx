//! omfx fork-owned direct provider model catalog.
//!
//! The Vercel gateway catalog is the only source the model picker and the
//! capability resolver see. When a model routes directly to a provider
//! (see registry.zig and provider_credentials.zig), that provider offers
//! models the gateway does not list. This module asks the provider for its
//! own model list and returns it as gateway catalog entries, so a caller
//! can merge both lists without a second entry type.
//!
//! The transport is an injected interface. Production code uses
//! `default_transport`; tests script the responses.

const std = @import("std");
const build_options = @import("build_options");
const io_mod = @import("../shared/io.zig");
const model_catalog = @import("../gateway/model_catalog.zig");
const model_capabilities = @import("../config/model_capabilities.zig");
const provider_credentials = @import("provider_credentials.zig");
const registry = @import("registry.zig");

const Allocator = std.mem.Allocator;

/// Largest model list accepted from a provider.
pub const response_max_bytes: usize = 4 * 1024 * 1024;

const user_agent = "fx/" ++ build_options.app_version;

pub const Error = error{
    /// Transport reached the provider but the provider refused the request.
    CatalogRequestFailed,
    /// The provider answered with a body this module cannot read.
    CatalogMalformed,
    /// The provider answered with more bytes than `response_max_bytes`.
    CatalogResponseTooLarge,
};

pub const HttpResult = struct {
    status: u16,
    /// Owned bytes allocated with the allocator passed to `HttpTransport.get`.
    body: []u8,

    pub fn deinit(self: *HttpResult, alloc: Allocator) void {
        alloc.free(self.body);
        self.* = .{ .status = 0, .body = &.{} };
    }
};

/// `oauth_transport.Request` carries no extra headers, and this path needs
/// two, so the module owns a smaller transport of its own.
pub const HttpGetFn = *const fn (
    ?*anyopaque,
    Allocator,
    url: []const u8,
    bearer: ?[]const u8,
    account_id: ?[]const u8,
) anyerror!HttpResult;

pub const HttpTransport = struct {
    context: ?*anyopaque = null,
    get_fn: HttpGetFn,

    pub fn get(
        self: HttpTransport,
        alloc: Allocator,
        url: []const u8,
        bearer: ?[]const u8,
        account_id: ?[]const u8,
    ) !HttpResult {
        return self.get_fn(self.context, alloc, url, bearer, account_id);
    }
};

/// Fetches the models the provider offers for this credential.
/// The caller owns the result and frees it with
/// `model_catalog.freeModelCatalog`.
pub fn fetchModels(
    alloc: Allocator,
    transport: HttpTransport,
    def: *const registry.Def,
    credential: *const provider_credentials.Credential,
) !std.ArrayList(model_catalog.ModelCatalogEntry) {
    const url = try modelsUrl(alloc, def, credential);
    defer alloc.free(url);

    const account_id: ?[]const u8 = switch (credential.kind) {
        .api_key => null,
        .oauth => credential.account_id,
    };

    var result = try transport.get(alloc, url, credential.token, account_id);
    defer result.deinit(alloc);
    if (result.status != 200) return Error.CatalogRequestFailed;

    return parseModels(alloc, def, result.body);
}

/// Both transports read the model list from `<base>/models`. The ChatGPT
/// Codex backend accepts a plain GET there with the bearer and the
/// account-id header.
fn modelsUrl(
    alloc: Allocator,
    def: *const registry.Def,
    credential: *const provider_credentials.Credential,
) ![]u8 {
    const base = registry.baseUrl(def, credential.kind == .oauth);
    const trimmed = std.mem.trimEnd(u8, base, "/");
    return std.fmt.allocPrint(alloc, "{s}/models", .{trimmed});
}

fn parseModels(
    alloc: Allocator,
    def: *const registry.Def,
    body: []const u8,
) !std.ArrayList(model_catalog.ModelCatalogEntry) {
    var parsed = std.json.parseFromSlice(std.json.Value, alloc, body, .{}) catch
        return Error.CatalogMalformed;
    defer parsed.deinit();

    const items = try modelArray(parsed.value);

    var entries: std.ArrayList(model_catalog.ModelCatalogEntry) = .empty;
    errdefer model_catalog.freeModelCatalog(alloc, &entries);

    for (items) |item| {
        const raw_id = modelId(item) orelse continue;
        if (raw_id.len == 0) continue;
        if (!isChatModel(def, raw_id)) continue;

        const id = try std.fmt.allocPrint(alloc, "{s}{s}", .{ def.model_prefix, raw_id });
        errdefer alloc.free(id);
        if (containsId(entries.items, id)) {
            alloc.free(id);
            continue;
        }

        const model_type = try alloc.dupe(u8, "language");
        errdefer alloc.free(model_type);

        try entries.append(alloc, .{
            .id = id,
            .model_type = model_type,
            .has_tool_use = true,
            .has_vision = hasVision(def, raw_id),
            .context_window = model_capabilities.contextWindowSize(id) orelse 0,
        });
    }

    std.mem.sort(model_catalog.ModelCatalogEntry, entries.items, {}, idDescending);
    return entries;
}

/// Accepts the OpenAI `{"data":[...]}` shape, a bare `{"models":[...]}`,
/// and a top-level array.
fn modelArray(root: std.json.Value) ![]const std.json.Value {
    switch (root) {
        .array => |array| return array.items,
        .object => |object| {
            if (object.get("data")) |data| {
                if (data == .array) return data.array.items;
            }
            if (object.get("models")) |models| {
                if (models == .array) return models.array.items;
            }
            return Error.CatalogMalformed;
        },
        else => return Error.CatalogMalformed,
    }
}

fn modelId(item: std.json.Value) ?[]const u8 {
    return switch (item) {
        .string => |value| value,
        .object => |object| blk: {
            const id = object.get("id") orelse break :blk null;
            break :blk if (id == .string) id.string else null;
        },
        else => null,
    };
}

const openai_non_chat_markers = [_][]const u8{
    "embedding",
    "whisper",
    "tts",
    "dall-e",
    "davinci",
    "babbage",
    "moderation",
    "audio",
    "realtime",
    "image",
    "transcribe",
    "search",
    "similarity",
};

fn isChatModel(def: *const registry.Def, raw_id: []const u8) bool {
    if (std.mem.eql(u8, def.key, "xai")) {
        return startsWithIgnoreCase(raw_id, "grok");
    }
    for (openai_non_chat_markers) |marker| {
        if (containsIgnoreCase(raw_id, marker)) return false;
    }
    return true;
}

/// OpenAI chat models all read images. For xAI only grok-4 and later do,
/// so an id without a readable major version stays false.
fn hasVision(def: *const registry.Def, raw_id: []const u8) bool {
    if (!std.mem.eql(u8, def.key, "xai")) return true;
    const major = grokMajorVersion(raw_id) orelse return false;
    return major >= 4;
}

fn grokMajorVersion(raw_id: []const u8) ?u32 {
    const prefix = "grok-";
    if (!startsWithIgnoreCase(raw_id, prefix)) return null;
    const rest = raw_id[prefix.len..];
    var digits: usize = 0;
    while (digits < rest.len and std.ascii.isDigit(rest[digits])) digits += 1;
    if (digits == 0) return null;
    return std.fmt.parseInt(u32, rest[0..digits], 10) catch null;
}

fn containsId(entries: []const model_catalog.ModelCatalogEntry, id: []const u8) bool {
    for (entries) |entry| {
        if (std.mem.eql(u8, entry.id, id)) return true;
    }
    return false;
}

/// Descending id order puts the newer version of a family first.
fn idDescending(
    _: void,
    a: model_catalog.ModelCatalogEntry,
    b: model_catalog.ModelCatalogEntry,
) bool {
    return std.mem.order(u8, a.id, b.id) == .gt;
}

fn startsWithIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len > haystack.len) return false;
    return std.ascii.eqlIgnoreCase(haystack[0..needle.len], needle);
}

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;
    var i: usize = 0;
    while (i <= haystack.len - needle.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(haystack[i .. i + needle.len], needle)) return true;
    }
    return false;
}

fn httpGet(
    _: ?*anyopaque,
    alloc: Allocator,
    url: []const u8,
    bearer: ?[]const u8,
    account_id: ?[]const u8,
) anyerror!HttpResult {
    var client: std.http.Client = .{ .allocator = alloc, .io = io_mod.getIo() };
    defer client.deinit();

    var auth_header: ?[]u8 = null;
    defer if (auth_header) |value| alloc.free(value);

    var headers: std.http.Client.Request.Headers = .{
        .user_agent = .{ .override = user_agent },
        .accept_encoding = .omit,
    };
    if (bearer) |token| {
        auth_header = try std.fmt.allocPrint(alloc, "Bearer {s}", .{token});
        headers.authorization = .{ .override = auth_header.? };
    }

    var extra_buf: [1]std.http.Header = undefined;
    var extra: []const std.http.Header = extra_buf[0..0];
    if (account_id) |id| {
        extra_buf[0] = .{ .name = "ChatGPT-Account-ID", .value = id };
        extra = extra_buf[0..1];
    }

    const response_buffer = try alloc.alloc(u8, response_max_bytes + 1);
    defer alloc.free(response_buffer);
    var response_writer = std.Io.Writer.fixed(response_buffer);

    const result = client.fetch(.{
        .location = .{ .url = url },
        .method = .GET,
        .headers = headers,
        .extra_headers = extra,
        .response_writer = &response_writer,
    }) catch |err| switch (err) {
        error.WriteFailed => return Error.CatalogResponseTooLarge,
        else => return err,
    };

    const body = response_writer.buffered();
    if (body.len > response_max_bytes) return Error.CatalogResponseTooLarge;

    return .{
        .status = @intFromEnum(result.status),
        .body = try alloc.dupe(u8, body),
    };
}

pub const default_transport = HttpTransport{ .get_fn = httpGet };

const FakeTransport = struct {
    status: u16 = 200,
    body: []const u8 = "{}",
    /// `fetchModels` owns the url and frees it, so the fake keeps a copy.
    url_buf: [256]u8 = undefined,
    url_len: usize = 0,
    seen_bearer: ?[]const u8 = null,
    seen_account_id: ?[]const u8 = null,
    calls: usize = 0,

    fn get(
        context: ?*anyopaque,
        alloc: Allocator,
        url: []const u8,
        bearer: ?[]const u8,
        account_id: ?[]const u8,
    ) anyerror!HttpResult {
        const self: *FakeTransport = @ptrCast(@alignCast(context.?));
        self.calls += 1;
        self.url_len = @min(url.len, self.url_buf.len);
        @memcpy(self.url_buf[0..self.url_len], url[0..self.url_len]);
        self.seen_bearer = bearer;
        self.seen_account_id = account_id;
        return .{ .status = self.status, .body = try alloc.dupe(u8, self.body) };
    }

    fn seenUrl(self: *const FakeTransport) []const u8 {
        return self.url_buf[0..self.url_len];
    }

    fn transport(self: *FakeTransport) HttpTransport {
        return .{ .context = self, .get_fn = FakeTransport.get };
    }
};

fn testCredential(kind: provider_credentials.Kind) provider_credentials.Credential {
    return .{ .kind = kind, .token = @constCast("token-1") };
}

test "openai data shape parses, filters, and prefixes ids" {
    const alloc = std.testing.allocator;
    var fake = FakeTransport{ .body =
        \\{"data":[
        \\  {"id":"gpt-5.2-codex"},
        \\  {"id":"text-embedding-3-large"},
        \\  {"id":"whisper-1"},
        \\  {"id":"dall-e-3"},
        \\  {"id":"gpt-4o-realtime-preview"},
        \\  {"id":"gpt-4o-audio-preview"},
        \\  {"id":"o3"},
        \\  {"no_id":true},
        \\  {"id":123}
        \\]}
    };
    var credential = testCredential(.api_key);
    var entries = try fetchModels(alloc, fake.transport(), registry.byKey("openai").?, &credential);
    defer model_catalog.freeModelCatalog(alloc, &entries);

    try std.testing.expectEqual(@as(usize, 2), entries.items.len);
    try std.testing.expectEqualStrings("openai/o3", entries.items[0].id);
    try std.testing.expectEqualStrings("openai/gpt-5.2-codex", entries.items[1].id);
    try std.testing.expectEqualStrings("language", entries.items[0].model_type);
    try std.testing.expect(entries.items[0].has_tool_use);
    try std.testing.expect(entries.items[0].has_vision);
    try std.testing.expectEqual(@as(u32, 200_000), entries.items[0].context_window);
    try std.testing.expectEqual(@as(u32, 256_000), entries.items[1].context_window);
    try std.testing.expectEqualStrings(
        "https://api.openai.com/v1/models",
        fake.seenUrl(),
    );
}

test "array of strings and models key are accepted" {
    const alloc = std.testing.allocator;
    var credential = testCredential(.api_key);

    var bare = FakeTransport{ .body = "[\"gpt-5.2\",\"tts-1\",\"\",42]" };
    var from_array = try fetchModels(alloc, bare.transport(), registry.byKey("openai").?, &credential);
    defer model_catalog.freeModelCatalog(alloc, &from_array);
    try std.testing.expectEqual(@as(usize, 1), from_array.items.len);
    try std.testing.expectEqualStrings("openai/gpt-5.2", from_array.items[0].id);

    var wrapped = FakeTransport{ .body = "{\"models\":[{\"id\":\"gpt-5.2\"}]}" };
    var from_object = try fetchModels(alloc, wrapped.transport(), registry.byKey("openai").?, &credential);
    defer model_catalog.freeModelCatalog(alloc, &from_object);
    try std.testing.expectEqual(@as(usize, 1), from_object.items.len);
    try std.testing.expectEqualStrings("openai/gpt-5.2", from_object.items[0].id);
}

test "xai keeps only grok ids and gates vision on the major version" {
    const alloc = std.testing.allocator;
    var fake = FakeTransport{ .body =
        \\{"data":[
        \\  {"id":"grok-4-fast"},
        \\  {"id":"grok-3-mini"},
        \\  {"id":"grok-image-1"},
        \\  {"id":"llama-3-70b"}
        \\]}
    };
    var credential = testCredential(.api_key);
    var entries = try fetchModels(alloc, fake.transport(), registry.byKey("xai").?, &credential);
    defer model_catalog.freeModelCatalog(alloc, &entries);

    try std.testing.expectEqual(@as(usize, 3), entries.items.len);
    try std.testing.expectEqualStrings("xai/grok-image-1", entries.items[0].id);
    try std.testing.expectEqualStrings("xai/grok-4-fast", entries.items[1].id);
    try std.testing.expectEqualStrings("xai/grok-3-mini", entries.items[2].id);
    try std.testing.expect(!entries.items[0].has_vision);
    try std.testing.expect(entries.items[1].has_vision);
    try std.testing.expect(!entries.items[2].has_vision);
    try std.testing.expectEqual(@as(u32, 131_072), entries.items[1].context_window);
}

test "repeated ids collapse to one entry" {
    const alloc = std.testing.allocator;
    var fake = FakeTransport{
        .body = "{\"data\":[{\"id\":\"gpt-5.2\"},{\"id\":\"gpt-5.2\"},{\"id\":\"gpt-5.1\"}]}",
    };
    var credential = testCredential(.api_key);
    var entries = try fetchModels(alloc, fake.transport(), registry.byKey("openai").?, &credential);
    defer model_catalog.freeModelCatalog(alloc, &entries);

    try std.testing.expectEqual(@as(usize, 2), entries.items.len);
    try std.testing.expectEqualStrings("openai/gpt-5.2", entries.items[0].id);
    try std.testing.expectEqualStrings("openai/gpt-5.1", entries.items[1].id);
}

test "a refused request and a broken body both fail" {
    const alloc = std.testing.allocator;
    var credential = testCredential(.api_key);

    var refused = FakeTransport{ .status = 401, .body = "{\"error\":\"no\"}" };
    try std.testing.expectError(
        Error.CatalogRequestFailed,
        fetchModels(alloc, refused.transport(), registry.byKey("openai").?, &credential),
    );

    var broken = FakeTransport{ .body = "not json" };
    try std.testing.expectError(
        Error.CatalogMalformed,
        fetchModels(alloc, broken.transport(), registry.byKey("openai").?, &credential),
    );

    var unusable = FakeTransport{ .body = "{\"data\":\"gpt-5.2\"}" };
    try std.testing.expectError(
        Error.CatalogMalformed,
        fetchModels(alloc, unusable.transport(), registry.byKey("openai").?, &credential),
    );
}

test "oauth openai sends the bearer and the account id to the codex backend" {
    const alloc = std.testing.allocator;
    var fake = FakeTransport{ .body = "{\"data\":[{\"id\":\"gpt-5.2-codex\"}]}" };
    var credential = provider_credentials.Credential{
        .kind = .oauth,
        .token = @constCast("oauth-token"),
        .account_id = @constCast("acct-42"),
    };
    var entries = try fetchModels(alloc, fake.transport(), registry.byKey("openai").?, &credential);
    defer model_catalog.freeModelCatalog(alloc, &entries);

    try std.testing.expectEqualStrings(
        "https://chatgpt.com/backend-api/codex/models",
        fake.seenUrl(),
    );
    try std.testing.expectEqualStrings("oauth-token", fake.seen_bearer.?);
    try std.testing.expectEqualStrings("acct-42", fake.seen_account_id.?);
}

test "xai oauth sends the bearer without an account id" {
    const alloc = std.testing.allocator;
    var fake = FakeTransport{ .body = "{\"data\":[{\"id\":\"grok-4\"}]}" };
    var credential = testCredential(.oauth);
    var entries = try fetchModels(alloc, fake.transport(), registry.byKey("xai").?, &credential);
    defer model_catalog.freeModelCatalog(alloc, &entries);

    try std.testing.expectEqualStrings("https://api.x.ai/v1/models", fake.seenUrl());
    try std.testing.expectEqualStrings("token-1", fake.seen_bearer.?);
    try std.testing.expect(fake.seen_account_id == null);
}

test "an api key credential never leaks an account id header" {
    const alloc = std.testing.allocator;
    var fake = FakeTransport{ .body = "{\"data\":[{\"id\":\"gpt-5.2\"}]}" };
    var credential = provider_credentials.Credential{
        .kind = .api_key,
        .token = @constCast("sk-test"),
        .account_id = @constCast("acct-42"),
    };
    var entries = try fetchModels(alloc, fake.transport(), registry.byKey("openai").?, &credential);
    defer model_catalog.freeModelCatalog(alloc, &entries);

    try std.testing.expect(fake.seen_account_id == null);
    try std.testing.expectEqualStrings("sk-test", fake.seen_bearer.?);
}
