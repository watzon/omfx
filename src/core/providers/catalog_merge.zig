//! omfx fork-owned catalog merge.
//!
//! Appends models from credentialed direct providers (openai, xai) to the
//! gateway model catalog so the picker, `fx models`, and capability
//! resolution see them. Direct fetch failures never break the gateway
//! catalog; a gateway failure still surfaces direct models when any exist.

const std = @import("std");
const oauth_transport = @import("../auth/oauth_transport.zig");
const model_catalog = @import("../gateway/model_catalog.zig");
const debug_trace = @import("../shared/debug_trace.zig");
const direct_model_catalog = @import("direct_model_catalog.zig");
const provider_credentials = @import("provider_credentials.zig");
const registry = @import("registry.zig");

const Allocator = std.mem.Allocator;

/// Fetches every credentialed direct provider's models. Per-provider
/// failures are logged and skipped.
fn collectDirectEntries(
    alloc: Allocator,
    transport: direct_model_catalog.HttpTransport,
    oauth_http: oauth_transport.Provider,
) Allocator.Error!std.ArrayList(model_catalog.ModelCatalogEntry) {
    var entries: std.ArrayList(model_catalog.ModelCatalogEntry) = .empty;
    errdefer model_catalog.freeModelCatalog(alloc, &entries);

    for (&registry.defs) |*def| {
        if (provider_credentials.preferredKind(def) == null) continue;
        var credential = provider_credentials.resolve(alloc, oauth_http, def) catch |err| {
            if (err == error.OutOfMemory) return error.OutOfMemory;
            debug_trace.logf("provider", "direct catalog credential failed provider={s} err={s}", .{ def.key, @errorName(err) });
            continue;
        };
        defer credential.deinit(alloc);
        var fetched = direct_model_catalog.fetchModels(alloc, transport, def, &credential) catch |err| {
            if (err == error.OutOfMemory) return error.OutOfMemory;
            debug_trace.logf("provider", "direct catalog fetch failed provider={s} err={s}", .{ def.key, @errorName(err) });
            continue;
        };
        defer fetched.deinit(alloc);
        entries.appendSlice(alloc, fetched.items) catch |err| {
            // Entries not moved yet stay owned by `fetched`; free them here
            // because deinit releases only the list storage.
            model_catalog.freeModelCatalog(alloc, &fetched);
            fetched = .empty;
            return err;
        };
        fetched.clearRetainingCapacity();
    }
    return entries;
}

fn catalogContains(entries: []const model_catalog.ModelCatalogEntry, id: []const u8) bool {
    for (entries) |entry| {
        if (std.mem.eql(u8, entry.id, id)) return true;
    }
    return false;
}

/// Takes ownership of `gateway_result` and returns the merged result.
pub fn mergeDirectProviders(
    alloc: Allocator,
    transport: direct_model_catalog.HttpTransport,
    oauth_http: oauth_transport.Provider,
    gateway_result: model_catalog.ProviderResult,
) Allocator.Error!model_catalog.ProviderResult {
    // Let the caller's anonymous retry handle rejected gateway credentials;
    // the retried call comes back through here.
    if (gateway_result == .failure and gateway_result.failure.category == .authentication) {
        return gateway_result;
    }

    var direct = try collectDirectEntries(alloc, transport, oauth_http);
    if (direct.items.len == 0) {
        direct.deinit(alloc);
        return gateway_result;
    }

    switch (gateway_result) {
        .catalog => |gateway_catalog| {
            var merged = gateway_catalog;
            errdefer model_catalog.freeModelCatalog(alloc, &merged);
            defer {
                model_catalog.freeModelCatalog(alloc, &direct);
            }
            var index: usize = 0;
            while (index < direct.items.len) {
                if (catalogContains(merged.items, direct.items[index].id)) {
                    index += 1;
                    continue;
                }
                try merged.append(alloc, direct.items[index]);
                _ = direct.swapRemove(index);
            }
            return .{ .catalog = merged };
        },
        .failure => |failure| {
            debug_trace.logf("provider", "gateway catalog unavailable category={t}; serving direct provider models only", .{failure.category});
            return .{ .catalog = direct };
        },
    }
}

test "gateway authentication failures pass through for the anonymous retry" {
    const result = try mergeDirectProviders(
        std.testing.allocator,
        direct_model_catalog.default_transport,
        .{ .execute_fn = undefined },
        .{ .failure = .{ .category = .authentication, .http_status = .unauthorized } },
    );
    try std.testing.expect(result == .failure);
    try std.testing.expectEqual(model_catalog.FailureCategory.authentication, result.failure.category);
}
