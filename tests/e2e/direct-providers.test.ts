// omfx fork-owned suite: direct provider routing (openai/, xai/) around the
// Vercel AI Gateway, using loopback fake providers and env credentials.
import { afterAll, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { runFx } from "../evals/eval-helpers";

const TIMEOUT = 30_000;

type CapturedRequest = {
  path: string;
  authorization: string;
  body: any;
};

type FakeProvider = {
  url: string;
  requests: CapturedRequest[];
  stop: () => void;
};

function sse(events: unknown[]): string {
  return (
    events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("") +
    "data: [DONE]\n\n"
  );
}

function startFakeProvider(
  respond: (request: CapturedRequest, index: number) => unknown[],
  models: string[] = [],
): FakeProvider {
  const requests: CapturedRequest[] = [];
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    async fetch(req) {
      if (req.method === "GET") {
        return Response.json({ data: models.map((id) => ({ id })) });
      }
      const captured: CapturedRequest = {
        path: new URL(req.url).pathname,
        authorization: req.headers.get("authorization") ?? "",
        body: await req.json(),
      };
      requests.push(captured);
      return new Response(sse(respond(captured, requests.length - 1)), {
        headers: { "content-type": "text/event-stream" },
      });
    },
  });
  return {
    url: `http://127.0.0.1:${server.port}`,
    requests,
    stop: () => server.stop(true),
  };
}

const cleanups: Array<() => void> = [];
afterAll(() => {
  for (const cleanup of cleanups) cleanup();
});

function tempHome(): string {
  const home = mkdtempSync(join(tmpdir(), "fx-e2e-direct-providers-"));
  cleanups.push(() => rmSync(home, { recursive: true, force: true }));
  return home;
}

const baseEnv = {
  AI_GATEWAY_API_KEY: undefined,
  VERCEL_OIDC_TOKEN: undefined,
  OPENAI_API_KEY: undefined,
  XAI_API_KEY: undefined,
  FX_AUTO_UPGRADE: "0",
};

test(
  "xai models route to the chat completions wire with the env api key",
  async () => {
    const provider = startFakeProvider(() => [
      { id: "cmpl-1", choices: [{ index: 0, delta: { role: "assistant", content: "hello " } }] },
      { id: "cmpl-1", choices: [{ index: 0, delta: { content: "from fake xai" } }] },
      { id: "cmpl-1", choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
      { id: "cmpl-1", choices: [], usage: { prompt_tokens: 3, completion_tokens: 2 } },
    ]);
    cleanups.push(provider.stop);

    const result = await runFx(["ask", "--no-save", "--json", "say hello"], {
      env: {
        ...baseEnv,
        HOME: tempHome(),
        XAI_API_KEY: "test-xai-key",
        OMFX_XAI_BASE_URL: provider.url,
        FX_MODEL: "xai/grok-4",
      },
    });

    expect(result.code).toBe(0);
    const output = JSON.parse(result.stdout);
    expect(output.output).toBe("hello from fake xai");
    expect(output.model).toBe("xai/grok-4");

    expect(provider.requests).toHaveLength(1);
    const request = provider.requests[0]!;
    expect(request.path).toBe("/chat/completions");
    expect(request.authorization).toBe("Bearer test-xai-key");
    expect(request.body.model).toBe("grok-4");
    expect(request.body.stream).toBe(true);
    expect(request.body.messages[0].role).toBe("system");
    expect(request.body.tools[0].type).toBe("function");
    expect(request.body.tools[0].function.name).toBeString();
  },
  TIMEOUT,
);

test(
  "openai models route to the responses wire with the env api key",
  async () => {
    const provider = startFakeProvider(() => [
      { type: "response.created", response: { id: "resp-1" } },
      { type: "response.output_text.delta", delta: "hello from fake openai" },
      {
        type: "response.completed",
        response: { id: "resp-1", usage: { input_tokens: 3, output_tokens: 2 } },
      },
    ]);
    cleanups.push(provider.stop);

    const result = await runFx(["ask", "--no-save", "--json", "say hello"], {
      env: {
        ...baseEnv,
        HOME: tempHome(),
        OPENAI_API_KEY: "test-openai-key",
        OMFX_OPENAI_BASE_URL: provider.url,
        FX_MODEL: "openai/gpt-5.2",
      },
    });

    expect(result.code).toBe(0);
    const output = JSON.parse(result.stdout);
    expect(output.output).toBe("hello from fake openai");

    expect(provider.requests).toHaveLength(1);
    const request = provider.requests[0]!;
    expect(request.path).toBe("/responses");
    expect(request.authorization).toBe("Bearer test-openai-key");
    expect(request.body.model).toBe("gpt-5.2");
    expect(request.body.store).toBe(false);
    expect(request.body.instructions).toBeString();
    expect(request.body.tools[0].type).toBe("function");
    expect(request.body.tools[0].name).toBeString();
    expect(request.body.input[0].type).toBe("message");
  },
  TIMEOUT,
);

test(
  "a direct-provider tool call round trip executes and reports the result",
  async () => {
    const provider = startFakeProvider((_request, index) =>
      index === 0
        ? [
            { type: "response.created", response: { id: "resp-t1" } },
            {
              type: "response.output_item.added",
              item: { type: "function_call", call_id: "call_read1", name: "read_file" },
            },
            {
              type: "response.function_call_arguments.delta",
              delta: JSON.stringify({ path: "hello.txt" }),
            },
            {
              type: "response.output_item.done",
              item: {
                type: "function_call",
                call_id: "call_read1",
                name: "read_file",
                arguments: JSON.stringify({ path: "hello.txt" }),
              },
            },
            {
              type: "response.completed",
              response: { id: "resp-t1", usage: { input_tokens: 3, output_tokens: 2 } },
            },
          ]
        : [
            { type: "response.created", response: { id: "resp-t2" } },
            { type: "response.output_text.delta", delta: "file content confirmed" },
            {
              type: "response.completed",
              response: { id: "resp-t2", usage: { input_tokens: 5, output_tokens: 2 } },
            },
          ],
    );
    cleanups.push(provider.stop);

    const workspace = mkdtempSync(join(tmpdir(), "fx-e2e-direct-tools-"));
    cleanups.push(() => rmSync(workspace, { recursive: true, force: true }));
    mkdirSync(workspace, { recursive: true });
    writeFileSync(join(workspace, "hello.txt"), "TOOL-ROUNDTRIP-OK\n");

    const result = await runFx(
      ["ask", "--no-save", "--auto", "--json", "read hello.txt"],
      {
        cwd: workspace,
        env: {
          ...baseEnv,
          HOME: tempHome(),
          OPENAI_API_KEY: "test-openai-key",
          OMFX_OPENAI_BASE_URL: provider.url,
          FX_MODEL: "openai/gpt-5.2",
        },
      },
    );

    expect(result.code).toBe(0);
    const output = JSON.parse(result.stdout);
    expect(output.output).toBe("file content confirmed");
    expect(output.tool_calls).toEqual([{ name: "read_file", status: "success" }]);

    expect(provider.requests).toHaveLength(2);
    const followup = provider.requests[1]!.body;
    const call = followup.input.find((item: any) => item.type === "function_call");
    const callOutput = followup.input.find(
      (item: any) => item.type === "function_call_output",
    );
    expect(call.call_id).toBe("call_read1");
    expect(call.name).toBe("read_file");
    expect(callOutput.call_id).toBe("call_read1");
    expect(callOutput.output).toContain("TOOL-ROUNDTRIP-OK");
  },
  TIMEOUT,
);

test(
  "direct-provider models without a credential still require gateway auth",
  async () => {
    const result = await runFx(["ask", "--no-save", "--json", "hi"], {
      env: {
        ...baseEnv,
        HOME: tempHome(),
        FX_MODEL: "xai/grok-4",
      },
    });
    expect(result.code).toBe(1);
    const output = JSON.parse(result.stdout);
    expect(output.error).toBe("MissingCredentials");
  },
  TIMEOUT,
);

test(
  "gateway models ignore direct provider credentials",
  async () => {
    const result = await runFx(["ask", "--no-save", "--json", "hi"], {
      env: {
        ...baseEnv,
        HOME: tempHome(),
        XAI_API_KEY: "test-xai-key",
        FX_MODEL: "zai/glm-5.2",
      },
    });
    expect(result.code).toBe(1);
    const output = JSON.parse(result.stdout);
    expect(output.error).toBe("MissingCredentials");
  },
  TIMEOUT,
);

test(
  "image attachments encode as input_image parts on the responses wire",
  async () => {
    const provider = startFakeProvider(() => [
      { type: "response.created", response: { id: "resp-img" } },
      { type: "response.output_text.delta", delta: "a tiny square" },
      {
        type: "response.completed",
        response: { id: "resp-img", usage: { input_tokens: 9, output_tokens: 3 } },
      },
    ]);
    cleanups.push(provider.stop);

    const workspace = mkdtempSync(join(tmpdir(), "fx-e2e-direct-image-"));
    cleanups.push(() => rmSync(workspace, { recursive: true, force: true }));
    const png = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
      "base64",
    );
    writeFileSync(join(workspace, "pixel.png"), png);

    const result = await runFx(
      ["ask", "--no-save", "--json", "--image", "pixel.png", "describe this"],
      {
        cwd: workspace,
        env: {
          ...baseEnv,
          HOME: tempHome(),
          OPENAI_API_KEY: "test-openai-key",
          OMFX_OPENAI_BASE_URL: provider.url,
          FX_MODEL: "openai/gpt-5.2",
        },
      },
    );

    expect(result.code).toBe(0);
    expect(JSON.parse(result.stdout).output).toBe("a tiny square");
    expect(provider.requests).toHaveLength(1);
    const input = provider.requests[0]!.body.input;
    const parts = input
      .filter((item: any) => item.type === "message")
      .flatMap((item: any) => item.content ?? []);
    const image = parts.find((part: any) => part.type === "input_image");
    expect(image.image_url).toStartWith("data:image/png;base64,");
  },
  TIMEOUT,
);

test(
  "fx models merges direct provider models into the catalog",
  async () => {
    const provider = startFakeProvider(() => [], ["grok-e2e-test-model"]);
    cleanups.push(provider.stop);

    const result = await runFx(["models"], {
      env: {
        ...baseEnv,
        HOME: tempHome(),
        XAI_API_KEY: "test-xai-key",
        OMFX_XAI_BASE_URL: provider.url,
      },
    });

    expect(result.code).toBe(0);
    expect(result.stdout).toContain("xai/grok-e2e-test-model");
  },
  TIMEOUT,
);

test(
  "fx status reports direct provider credentials",
  async () => {
    const result = await runFx(["status", "--json"], {
      env: {
        ...baseEnv,
        HOME: tempHome(),
        XAI_API_KEY: "test-xai-key",
      },
    });
    expect(result.code).toBe(0);
    expect(JSON.parse(result.stdout).direct_providers).toBe("xai (api key)");
  },
  TIMEOUT,
);

test(
  "fx login rejects an unknown provider argument",
  async () => {
    const result = await runFx(["login", "not-a-provider"], {
      env: { ...baseEnv, HOME: tempHome() },
    });
    expect(result.code).not.toBe(0);
    expect(result.stderr).toContain("usage: fx login [openai|grok]");
  },
  TIMEOUT,
);

test(
  "fx logout for a provider without a session reports no saved login",
  async () => {
    const result = await runFx(["logout", "grok"], {
      env: { ...baseEnv, HOME: tempHome() },
    });
    expect(result.code).toBe(0);
    expect(result.stdout).toContain("No saved login for that provider.");
  },
  TIMEOUT,
);
