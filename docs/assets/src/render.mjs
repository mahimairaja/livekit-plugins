// Renders docs/assets/hero.gif and docs/assets/pipeline.png from hero.html and arch.html.
// Usage (from this folder): npm ci && python3 fetch_fonts.py && PW_EXE=<chromium path> node render.mjs
import { chromium } from "playwright";
import { mkdirSync, rmSync } from "fs";
import { execFileSync } from "child_process";

const b = await chromium.launch(process.env.PW_EXE ? { executablePath: process.env.PW_EXE } : {});
const url = (f) => "file://" + process.cwd() + "/" + f;

// Static "what's inside" diagram at 2x for sharp text.
let p = await b.newPage({ viewport: { width: 1280, height: 600 }, deviceScaleFactor: 2 });
await p.goto(url("arch.html"));
await p.waitForFunction(() => window.ready === true);
await (await p.$("#stage")).screenshot({ path: "../pipeline.png" });

// Animated hero: step the timeline frame by frame, then encode with ffmpeg.
p = await b.newPage({ viewport: { width: 1280, height: 640 }, deviceScaleFactor: 1 });
await p.goto(url("hero.html"));
await p.waitForFunction(() => window.ready === true);
rmSync("frames", { recursive: true, force: true });
mkdirSync("frames");
const fps = 15, dur = await p.evaluate(() => window.DURATION);
const el = await p.$("#stage");
for (let i = 0; i < Math.round(dur * fps); i++) {
  await p.evaluate((t) => window.setT(t), i / fps);
  await el.screenshot({ path: `frames/${String(i).padStart(4, "0")}.png` });
}
await b.close();
execFileSync("ffmpeg", ["-v", "error", "-y", "-framerate", String(fps), "-i", "frames/%04d.png", "-vf",
  "split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
  "-loop", "0", "../hero.gif"], { stdio: "inherit" });
rmSync("frames", { recursive: true, force: true });
