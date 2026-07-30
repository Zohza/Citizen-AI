/**
 * Vite plugin that clears the terminal log before each rebuild.
 * Internal Dala plugin — no-op stub.
 */
export default function clearLogPlugin() {
  return {
    name: "dala-clear-log",
    configureServer() {
      // No-op: terminal clearing is handled by the dev environment
    },
  };
}
