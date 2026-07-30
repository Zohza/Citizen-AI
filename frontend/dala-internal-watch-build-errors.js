/**
 * Internal Dala script — watches build.log for errors and prints them.
 * Stub: logs build errors from the log file.
 */
import { watch } from "fs";
import { createInterface } from "readline";

const BUILD_LOG = "./build.log";

console.log("[watch:errors] Watching", BUILD_LOG, "for build errors...");

try {
  watch(BUILD_LOG, (eventType) => {
    if (eventType === "change") {
      const rl = createInterface({
        input: process.stdin,
        output: process.stdout,
      });
      // Stub — in production this would tail and filter errors
    }
  });
} catch {
  console.log("[watch:errors] Build log not found yet. Waiting for build...");
}
