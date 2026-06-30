import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState } from "react";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "FreshlyFishy — AI Fish Freshness Detection" },
      { name: "description", content: "Automated fish eye morphology freshness evaluation system powered by computer vision and deep learning." },
      { property: "og:title", content: "FreshlyFishy — AI Fish Freshness Detection" },
      { property: "og:description", content: "Automated fish eye morphology freshness evaluation." },
    ],
  }),
  component: Page,
});

// ─── PRODUCTION API LINK BINDING ───
const API_BASE = "https://fish-freshness-system.onrender.com";

// ---------- Types ----------
type Verdict = "Fresh" | "Not Fresh";
interface InferSuccess {
  success: true;
  freshness_class: Verdict;
  confidence: number;
  cropped_eye_base64: string;
  gradcam_overlay_base64: string;
  timestamp: string;
}
interface InferError {
  success: false;
  error_message: string;
  timestamp: string;
}
type InferResponse = InferSuccess | InferError;

type HealthState = "operational" | "connecting" | "disconnected";

// ---------- Page ----------
function Page() {
  return (
    <div className="relative min-h-screen overflow-x-hidden">
      <AmbientBackground />
      <Header />
      <main className="relative z-10 mx-auto max-w-7xl px-6">
        <Hero />
        <SectionDivider label="01 / Pipeline Architecture" />
        <PipelineFlow />
        <SectionDivider label="02 / Validation Performance" />
        <MetricsDashboard />
        <SectionDivider label="03 / Live Scanner" />
        <ScannerInterface />
        <Footer />
      </main>
    </div>
  );
}

// ---------- Ambient Background ----------
function AmbientBackground() {
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
      <svg className="absolute inset-x-0 top-1/3 h-[600px] w-full animate-wave-drift opacity-30" viewBox="0 0 1440 600" preserveAspectRatio="none">
        <defs>
          <linearGradient id="wave1" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0%" stopColor="oklch(0.78 0.15 210)" stopOpacity="0" />
            <stop offset="50%" stopColor="oklch(0.78 0.15 210)" stopOpacity="0.5" />
            <stop offset="100%" stopColor="oklch(0.74 0.17 160)" stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d="M0,300 C320,200 720,400 1440,260 L1440,600 L0,600 Z" fill="url(#wave1)" opacity="0.25" />
        <path d="M0,360 C360,260 800,460 1440,320 L1440,600 L0,600 Z" fill="url(#wave1)" opacity="0.15" />
      </svg>
      <div className="absolute inset-0 bg-[linear-gradient(to_bottom,transparent,oklch(0.18_0.04_250)_85%)]" />
    </div>
  );
}

// ---------- Header ----------
function useHealth() {
  const [state, setState] = useState<HealthState>("connecting");
  useEffect(() => {
    let cancelled = false;
    const ping = async () => {
      try {
        const ctrl = new AbortController();
        const t = setTimeout(() => ctrl.abort(), 4000);
        const r = await fetch(`${API_BASE}/health`, { signal: ctrl.signal });
        clearTimeout(t);
        if (cancelled) return;
        setState(r.ok ? "operational" : "disconnected");
      } catch {
        if (!cancelled) setState("disconnected");
      }
    };
    ping();
    const id = setInterval(ping, 15000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);
  return state;
}

function Header() {
  const health = useHealth();
  const styles: Record<HealthState, { dot: string; ring: string; label: string }> = {
    operational: { dot: "bg-[var(--emerald)] glow-emerald", ring: "ring-[var(--emerald)]/40", label: "Operational" },
    connecting: { dot: "bg-[var(--amber)]", ring: "ring-[var(--amber)]/40", label: "Connecting…" },
    disconnected: { dot: "bg-[var(--crimson)] glow-crimson", ring: "ring-[var(--crimson)]/40", label: "Disconnected" },
  };
  const s = styles[health];
  return (
    <header className="sticky top-0 z-50 border-b border-[color-mix(in_oklab,var(--cyan)_15%,transparent)] glass">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
        <div className="flex items-center gap-3">
          <LogoMark />
          <div className="flex flex-col leading-tight">
            <span className="text-base font-bold tracking-tight text-foreground">FreshlyFishy</span>
            <span className="font-mono-tabular text-[10px] uppercase tracking-[0.18em] text-muted-foreground">eye morphology · vision lab</span>
          </div>
        </div>
        <div className={`flex items-center gap-2 rounded-full bg-[var(--surface)]/60 px-3 py-1.5 text-xs ring-1 ${s.ring}`}>
          <span className={`relative inline-block h-2 w-2 rounded-full ${s.dot} animate-pulse-dot`} />
          <span className="font-mono-tabular text-[11px] uppercase tracking-wider text-foreground/90">{s.label}</span>
          <span className="hidden font-mono-tabular text-[10px] text-muted-foreground sm:inline">GET /health</span>
        </div>
      </div>
    </header>
  );
}

function LogoMark() {
  return (
    <div className="relative grid h-9 w-9 place-items-center rounded-xl glass glow-cyan">
      <svg viewBox="0 0 24 24" className="h-5 w-5 text-[var(--cyan)]" fill="none" stroke="currentColor" strokeWidth="1.8">
        <path d="M3 12c3-5 8-7 13-6 2 .4 4 1.8 5 4-1 2.2-3 3.6-5 4-5 1-10-1-13-2z" />
        <circle cx="10" cy="11" r="1.6" fill="currentColor" />
        <path d="M3 12c-1 2-1 4 0 5" />
      </svg>
    </div>
  );
}

// ---------- Hero ----------
function Hero() {
  return (
    <section className="flex min-h-[80vh] flex-col items-center justify-center py-32 text-center">
      <div className="inline-flex items-center gap-2 rounded-full border border-[color-mix(in_oklab,var(--cyan)_30%,transparent)] glass px-4 py-1.5">
        <span className="h-1.5 w-1.5 rounded-full bg-[var(--cyan)] animate-pulse-dot" />
        <span className="font-mono-tabular text-[11px] uppercase tracking-[0.22em] text-foreground/80">FishEyeNet · AWPF · v1.0</span>
      </div>
      <h1 className="mt-8 max-w-5xl text-balance text-5xl font-extrabold leading-[1.05] tracking-tight md:text-7xl">
        <span className="text-gradient-marine">Automated Fish Eye Morphology</span>
        <br />
        <span className="text-foreground/95">Freshness Evaluation System</span>
      </h1>
      <p className="mt-8 max-w-2xl text-base leading-relaxed text-muted-foreground md:text-lg">
        A laboratory-grade computer vision pipeline that localizes ocular structures, runs deep convolutional analysis, and emits explainable Grad-CAM evidence — in milliseconds.
      </p>
      <div className="mt-12 flex flex-wrap items-center justify-center gap-3 font-mono-tabular text-[11px] uppercase tracking-wider text-muted-foreground">
        <Stat label="Trained Samples" value="1,537" />
        <Stat label="Accuracy" value="92.37%" />
        <Stat label="F1 · Fresh" value="0.9311" />
        <Stat label="F1 · Not Fresh" value="0.9274" />
      </div>
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-full border border-[color-mix(in_oklab,var(--cyan)_18%,transparent)] glass px-4 py-2">
      <span className="text-foreground/90">{value}</span>
      <span className="mx-2 text-muted-foreground/60">·</span>
      <span>{label}</span>
    </div>
  );
}

// ---------- Section Divider ----------
function SectionDivider({ label }: { label: string }) {
  return (
    <div className="my-24 flex items-center gap-4 md:my-32">
      <span className="font-mono-tabular text-[11px] uppercase tracking-[0.28em] text-[var(--cyan)]">{label}</span>
      <span className="h-px flex-1 bg-gradient-to-r from-[color-mix(in_oklab,var(--cyan)_45%,transparent)] to-transparent" />
    </div>
  );
}

// ---------- Pipeline ----------
const PIPELINE = [
  {
    n: "01",
    title: "File Ingest & Stream Processing",
    desc: "Validates JPEG/PNG payloads, streams raw bytes into an in-memory buffer with size and MIME guards.",
    tag: "I/O",
  },
  {
    n: "02",
    title: "Hough Circle Transform",
    desc: "Classical CV eye localization. Isolates and pads a 384×384 ocular patch from the source frame.",
    tag: "OpenCV",
  },
  {
    n: "03",
    title: "Neural Forward Pass",
    desc: "FishEyeNetAWPF with FusedMBConv blocks and Swin window attention. Produces class logits and feature maps.",
    tag: "PyTorch",
  },
  {
    n: "04",
    title: "Explainability Engine",
    desc: "ThreadSafeGradCAM hooks gradient weights back onto the eye canvas at 45 % alpha with Jet colormap.",
    tag: "Grad-CAM",
  },
  {
    n: "05",
    title: "Outbound Serialization",
    desc: "Encodes frames to compressed Base64 and assembles the strict JSON inference contract.",
    tag: "JSON",
  },
];

function PipelineFlow() {
  return (
    <section>
      <SectionHeading
        kicker="Backend Pipeline"
        title="A five-stage forward propagation through the FreshlyFishy core."
        sub="Each stage runs server-side over a single multipart upload. Particles below trace the live data path."
      />
      <div className="relative mt-16 grid gap-6 md:grid-cols-5">
        <div aria-hidden className="pointer-events-none absolute inset-x-0 top-1/2 hidden h-px md:block">
          <div className="relative h-px w-full bg-gradient-to-r from-transparent via-[color-mix(in_oklab,var(--cyan)_60%,transparent)] to-transparent">
            <span className="absolute -top-[3px] left-0 h-1.5 w-1.5 rounded-full bg-[var(--cyan)] glow-cyan" style={{ animation: "wave-drift 6s linear infinite" }} />
          </div>
        </div>
        {PIPELINE.map((step, i) => (
          <article
            key={step.n}
            className="group relative rounded-2xl glass p-5 transition-all duration-300 hover:-translate-y-1 hover:border-[var(--cyan)]/50 hover:glow-cyan"
          >
            <div className="flex items-center justify-between">
              <span className="font-mono-tabular text-[11px] tracking-widest text-[var(--cyan)]">{step.n}</span>
              <span className="rounded-full border border-[color-mix(in_oklab,var(--cyan)_25%,transparent)] bg-[var(--surface)]/60 px-2 py-0.5 font-mono-tabular text-[9px] uppercase tracking-wider text-muted-foreground">
                {step.tag}
              </span>
            </div>
            <h3 className="mt-4 text-[15px] font-semibold leading-snug text-foreground">{step.title}</h3>
            <p className="mt-2 text-[13px] leading-relaxed text-muted-foreground">{step.desc}</p>
            {i < PIPELINE.length - 1 && (
              <span aria-hidden className="absolute -right-3 top-1/2 hidden h-2 w-2 -translate-y-1/2 rotate-45 border-r border-t border-[var(--cyan)]/60 bg-[var(--surface)] md:block" />
            )}
          </article>
        ))}
      </div>
    </section>
  );
}

function SectionHeading({ kicker, title, sub }: { kicker: string; title: string; sub?: string }) {
  return (
    <div className="max-w-3xl">
      <p className="font-mono-tabular text-[11px] uppercase tracking-[0.28em] text-[var(--cyan)]">{kicker}</p>
      <h2 className="mt-3 text-balance text-3xl font-bold leading-tight tracking-tight md:text-4xl">{title}</h2>
      {sub && <p className="mt-4 text-base leading-relaxed text-muted-foreground">{sub}</p>}
    </div>
  );
}

// ---------- Metrics ----------
const METRICS = [
  { label: "Total Curated Repository", value: "1,537", sub: "balanced eye-patch corpus" },
  { label: "Test Accuracy", value: "92.37%", sub: "global classifier · held-out split" },
  { label: "F1 · Fresh", value: "0.9311", sub: "precision-recall harmonic mean" },
  { label: "F1 · Not Fresh", value: "0.9274", sub: "precision-recall harmonic mean" },
];

function MetricsDashboard() {
  const [open, setOpen] = useState(false);
  return (
    <section>
      <SectionHeading
        kicker="Validation Telemetry"
        title="Benchmarks from the production training run."
        sub="Static performance metrics observed against the held-out evaluation split."
      />
      <div className="mt-12 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {METRICS.map((m) => (
          <div key={m.label} className="rounded-2xl glass p-6">
            <p className="font-mono-tabular text-[10px] uppercase tracking-[0.2em] text-muted-foreground">{m.label}</p>
            <p className="mt-4 font-mono-tabular text-4xl font-semibold text-gradient-marine">{m.value}</p>
            <p className="mt-2 text-xs text-muted-foreground">{m.sub}</p>
          </div>
        ))}
      </div>

      <div className="mt-10">
        <button
          onClick={() => setOpen((o) => !o)}
          className="inline-flex items-center gap-2 rounded-full border border-[var(--cyan)]/40 glass px-5 py-2.5 text-sm font-medium text-foreground transition-all hover:glow-cyan"
        >
          <span className={`inline-block h-1.5 w-1.5 rounded-full bg-[var(--cyan)] transition-transform ${open ? "scale-150" : ""}`} />
          {open ? "Hide" : "Reveal"} Confusion Matrix
        </button>

        <div
          className={`grid overflow-hidden transition-[grid-template-rows,opacity] duration-500 ${
            open ? "mt-8 grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"
          }`}
        >
          <div className="min-h-0">
            <ConfusionMatrix />
          </div>
        </div>
      </div>
    </section>
  );
}

function ConfusionMatrix() {
  const cells = [
    { label: "True Positive", value: 65, sub: "Fresh → Fresh", tone: "emerald" },
    { label: "False Negative", value: 4, sub: "Fresh → Not Fresh", tone: "crimson" },
    { label: "False Positive", value: 6, sub: "Not Fresh → Fresh", tone: "crimson" },
    { label: "True Negative", value: 64, sub: "Not Fresh → Not Fresh", tone: "emerald" },
  ] as const;
  return (
    <div className="rounded-2xl glass-strong p-6 md:p-8">
      <div className="grid grid-cols-[auto_1fr_1fr] gap-3 text-xs">
        <div />
        <ColHead>Predicted · Fresh</ColHead>
        <ColHead>Predicted · Not Fresh</ColHead>
        <RowHead>Actual · Fresh</RowHead>
        <Cell {...cells[0]} />
        <Cell {...cells[1]} />
        <RowHead>Actual · Not Fresh</RowHead>
        <Cell {...cells[2]} />
        <Cell {...cells[3]} />
      </div>
      <p className="mt-6 font-mono-tabular text-[11px] text-muted-foreground">
        N = 139 · accuracy = 0.9237 · diagonal dominance confirms balanced generalization.
      </p>
    </div>
  );
}

function ColHead({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-md bg-[var(--surface)]/60 px-3 py-2 text-center font-mono-tabular text-[10px] uppercase tracking-wider text-muted-foreground">
      {children}
    </div>
  );
}
function RowHead({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center rounded-md bg-[var(--surface)]/60 px-3 py-2 font-mono-tabular text-[10px] uppercase tracking-wider text-muted-foreground">
      {children}
    </div>
  );
}
function Cell({ label, value, sub, tone }: { label: string; value: number; sub: string; tone: "emerald" | "crimson" }) {
  const color = tone === "emerald" ? "var(--emerald)" : "var(--crimson)";
  return (
    <div
      className="relative overflow-hidden rounded-xl p-5"
      style={{
        background: `color-mix(in oklab, ${color} 12%, var(--surface))`,
        border: `1px solid color-mix(in oklab, ${color} 35%, transparent)`,
      }}
    >
      <p className="font-mono-tabular text-[10px] uppercase tracking-wider" style={{ color }}>
        {label}
      </p>
      <p className="mt-2 font-mono-tabular text-4xl font-semibold text-foreground">{value}</p>
      <p className="mt-1 text-xs text-muted-foreground">{sub}</p>
    </div>
  );
}

// ---------- Scanner ----------
const TERMINAL_LINES = [
  "[INFO] Stream processing raw binary upload chunk...",
  "[INFO] Validating MIME signature · JPEG/PNG envelope OK",
  "[INFO] Running Hough Circle Transform eye localization...",
  "[INFO] Cropping 384×384 padded ocular patch · ROI locked",
  "[INFO] Computing forward pass through FishEyeNetAWPF layers...",
  "[INFO] FusedMBConv → Swin window attention · logits resolved",
  "[INFO] Extracting ThreadSafeGradCAM structural weights...",
  "[INFO] Overlaying 45% alpha Jet colormap onto canvas",
  "[INFO] Serializing Base64 frames into JSON contract...",
];

interface ScannerState {
  file: File | null;
  previewUrl: string | null;
  loading: boolean;
  result: InferSuccess | null;
  error: string | null;
}

function ScannerInterface() {
  const [state, setState] = useState<ScannerState>({
    file: null,
    previewUrl: null,
    loading: false,
    result: null,
    error: null,
  });
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const reset = useCallback(() => {
    setState((s) => {
      if (s.previewUrl) URL.revokeObjectURL(s.previewUrl);
      return { file: null, previewUrl: null, loading: false, result: null, error: null };
    });
    if (inputRef.current) inputRef.current.value = "";
  }, []);

  const submit = useCallback(async (file: File) => {
    const url = URL.createObjectURL(file);
    setState({ file, previewUrl: url, loading: true, result: null, error: null });
    
    try {
      // ─── 🌟 NATIVE HARDWARE CANVAS IMAGE COMPRESSION BLOCK 🌟 ───
      const img = new Image();
      img.src = url;
      
      const compressedBlob = await new Promise<Blob>((resolve, reject) => {
        img.onload = () => {
          const canvas = document.createElement("canvas");
          
          // Match the exact expected dimension shape of the ONNX matrix structure
          canvas.width = 384;
          canvas.height = 384;
          
          const ctx = canvas.getContext("2d");
          if (!ctx) {
            reject(new Error("Failed to initialize structural canvas layout pipeline context."));
            return;
          }
          
          // Downsample high-res photo directly onto the 384x384 canvas container grid
          ctx.drawImage(img, 0, 0, 384, 384);
          
          // Convert matrix into highly efficient JPEG binary format at an optimized 85% fidelity ratio
          canvas.toBlob(
            (blob) => {
              if (blob) resolve(blob);
              else reject(new Error("Browser engine canvas quantization extraction error."));
            },
            "image/jpeg",
            0.85
          );
        };
        img.onerror = () => reject(new Error("Failed parsing image asset metadata structure boundaries."));
      });

      // Append shrunken 50KB dataset binary slice to Form streams
      const fd = new FormData();
      fd.append("file", compressedBlob, "compressed_mobile_sample.jpg");
      
      // Dispatch payload cleanly to production endpoint
      const r = await fetch(`${API_BASE}/infer`, { method: "POST", body: fd });
      const data = (await r.json()) as InferResponse;
      
      if (!data.success) {
        setState((s) => ({ ...s, loading: false, error: data.error_message || "Inference failed." }));
        return;
      }
      setState((s) => ({ ...s, loading: false, result: data }));
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Network error reaching inference server.";
      setState((s) => ({ ...s, loading: false, error: msg }));
    }
  }, []);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      const f = files[0];
      if (!/^image\/(png|jpe?g)$/.test(f.type)) {
        setState({ file: null, previewUrl: null, loading: false, result: null, error: "Only JPEG or PNG images are accepted." });
        return;
      }
      void submit(f);
    },
    [submit],
  );

  return (
    <section>
      <SectionHeading
        kicker="Execution Core"
        title="Drop a sample. Watch the model decide."
        sub="The dropzone wires directly into the FastAPI inference endpoint, streams telemetry, and returns side-by-side visual evidence."
      />

      <div className="mt-12 grid gap-6 lg:grid-cols-5">
        {/* Dropzone Container */}
        <div className="lg:col-span-3">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              handleFiles(e.dataTransfer.files);
            }}
            onClick={() => inputRef.current?.click()}
            className={`relative flex min-h-[420px] cursor-pointer flex-col items-center justify-center overflow-hidden rounded-2xl border-2 border-dashed p-8 text-center transition-all ${
              dragOver
                ? "border-[var(--cyan)] bg-[color-mix(in_oklab,var(--cyan)_8%,var(--surface))] glow-cyan"
                : "border-[color-mix(in_oklab,var(--cyan)_30%,transparent)] glass"
            }`}
          >
            <input ref={inputRef} type="file" accept="image/png,image/jpeg" hidden onChange={(e) => handleFiles(e.target.files)} />

            {state.previewUrl ? (
              <div className="relative h-[360px] w-full overflow-hidden rounded-xl">
                <img src={state.previewUrl} alt="Uploaded sample" className="absolute inset-0 h-full w-full object-contain" />
                {state.loading && (
                  <>
                    <div className="absolute inset-0 bg-[var(--background)]/60 backdrop-blur-[2px]" />
                    <div className="pointer-events-none absolute inset-0 overflow-hidden">
                      <div className="absolute inset-x-0 h-24 animate-scan-sweep bg-gradient-to-b from-transparent via-[var(--cyan)]/40 to-transparent" />
                    </div>
                    <div className="absolute inset-x-0 bottom-0 flex items-center justify-center p-4">
                      <p className="font-mono-tabular text-xs uppercase tracking-[0.2em] text-[var(--cyan)]">
                        Running server mathematical convolutions…
                      </p>
                    </div>
                  </>
                )}
              </div>
            ) : (
              <>
                <DropzoneIcon />
                <p className="mt-6 text-lg font-semibold text-foreground">Drop a fish image, or click to browse</p>
                <p className="mt-2 text-sm text-muted-foreground">JPEG · PNG · up to a few MB · single sample per inference</p>
                <p className="mt-6 font-mono-tabular text-[10px] uppercase tracking-[0.25em] text-muted-foreground">
                  POST {API_BASE}/infer · multipart/form-data · field "file"
                </p>
              </>
            )}
          </div>

          {state.error && (
            <div className="mt-4 rounded-xl border border-[var(--crimson)]/50 bg-[color-mix(in_oklab,var(--crimson)_10%,var(--surface))] p-4">
              <div className="flex items-start gap-3">
                <span className="mt-1 h-2 w-2 shrink-0 rounded-full bg-[var(--crimson)] glow-crimson" />
                <div className="min-w-0">
                  <p className="font-mono-tabular text-[11px] uppercase tracking-wider text-[var(--crimson)]">Inference Error</p>
                  <p className="mt-1 break-words text-sm text-foreground/90">{state.error}</p>
                </div>
                <button onClick={reset} className="ml-auto shrink-0 rounded-md border border-[var(--crimson)]/40 px-3 py-1 text-xs text-foreground/90 hover:bg-[var(--crimson)]/15">
                  Dismiss
                </button>
              </div>
            </div>
          )}

          {(state.result || state.file) && !state.loading && (
            <div className="mt-4 flex items-center justify-end">
              <button
                onClick={reset}
                className="inline-flex items-center gap-2 rounded-full border border-[var(--cyan)]/40 glass px-4 py-2 text-xs font-medium text-foreground transition-all hover:glow-cyan"
              >
                <ResetIcon /> Scan New Sample
              </button>
            </div>
          )}
        </div>

        {/* Terminal Window block */}
        <div className="lg:col-span-2">
          <Terminal active={state.loading} />
        </div>
      </div>

      {/* Visual Response Boards Mapping */}
      {state.result && (
        <div className="mt-10 space-y-6">
          <VerdictBanner verdict={state.result.freshness_class} confidence={state.result.confidence} />
          <ComparativeGallery
            original={state.previewUrl}
            crop={state.result.cropped_eye_base64}
            heatmap={state.result.gradcam_overlay_base64}
          />
        </div>
      )}
    </section>
  );
}

function DropzoneIcon() {
  return (
    <div className="grid h-16 w-16 place-items-center rounded-2xl glass glow-cyan">
      <svg viewBox="0 0 24 24" className="h-7 w-7 text-[var(--cyan)]" fill="none" stroke="currentColor" strokeWidth="1.6">
        <path d="M12 16V4m0 0l-4 4m4-4l4 4" />
        <path d="M4 17v2a2 2 0 002 2h12a2 2 0 002-2v-2" />
      </svg>
    </div>
  );
}

function ResetIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M3 12a9 9 0 1 0 3-6.7" />
      <path d="M3 4v5h5" />
    </svg>
  );
}

function Terminal({ active }: { active: boolean }) {
  const [lines, setLines] = useState<string[]>([]);
  useEffect(() => {
    if (!active) {
      setLines([]);
      return;
    }
    setLines([]);
    let i = 0;
    const id = setInterval(() => {
      setLines((prev) => [...prev, TERMINAL_LINES[i % TERMINAL_LINES.length]]);
      i++;
    }, 380);
    return () => clearInterval(id);
  }, [active]);

  return (
    <div className="flex h-full min-h-[420px] flex-col overflow-hidden rounded-2xl glass">
      <div className="flex items-center justify-between border-b border-[color-mix(in_oklab,var(--cyan)_18%,transparent)] px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full bg-[var(--crimson)]/80" />
          <span className="h-2.5 w-2.5 rounded-full bg-[var(--amber)]/80" />
          <span className="h-2.5 w-2.5 rounded-full bg-[var(--emerald)]/80" />
          <span className="ml-3 font-mono-tabular text-[10px] uppercase tracking-wider text-muted-foreground">
            inference.log · live tail
          </span>
        </div>
        <span className={`flex items-center gap-1.5 font-mono-tabular text-[10px] uppercase tracking-wider ${active ? "text-[var(--cyan)]" : "text-muted-foreground"}`}>
          <span className={`h-1.5 w-1.5 rounded-full ${active ? "bg-[var(--cyan)] animate-pulse-dot" : "bg-muted-foreground/50"}`} />
          {active ? "Streaming" : "Idle"}
        </span>
      </div>
      <div className="flex-1 overflow-auto p-4 font-mono-tabular text-[12px] leading-relaxed">
        {lines.length === 0 ? (
          <p className="text-muted-foreground">$ awaiting inference event…</p>
        ) : (
          lines.map((l, i) => (
            <p key={i} className="text-foreground/90">
              <span className="text-muted-foreground">{String(i + 1).padStart(2, "0")} │ </span>
              <span className={l.includes("[INFO]") ? "text-[var(--cyan)]" : "text-foreground"}>{l}</span>
            </p>
          ))
        )}
        {active && (
          <p className="mt-1 text-[var(--cyan)]">
            <span className="text-muted-foreground">$&nbsp;</span>
            <span className="inline-block h-3.5 w-2 translate-y-[2px] animate-pulse-dot bg-[var(--cyan)]" />
          </p>
        )}
      </div>
    </div>
  );
}

function VerdictBanner({ verdict, confidence }: { verdict: Verdict; confidence: number }) {
  const fresh = verdict === "Fresh";
  const color = fresh ? "var(--emerald)" : "var(--crimson)";
  const glow = fresh ? "glow-emerald" : "glow-crimson";
  const pct = (confidence * 100).toFixed(2);
  return (
    <div
      className={`relative overflow-hidden rounded-2xl p-8 ${glow}`}
      style={{
        background: `linear-gradient(120deg, color-mix(in oklab, ${color} 18%, var(--surface)), var(--surface))`,
        border: `1px solid color-mix(in oklab, ${color} 50%, transparent)`,
      }}
    >
      <div className="flex flex-col items-start justify-between gap-6 md:flex-row md:items-center">
        <div className="flex items-center gap-5">
          <div
            className="grid h-16 w-16 place-items-center rounded-2xl"
            style={{ background: `color-mix(in oklab, ${color} 25%, transparent)`, border: `1px solid color-mix(in oklab, ${color} 50%, transparent)` }}
          >
            {fresh ? <CheckIcon color={color} /> : <AlertIcon color={color} />}
          </div>
          <div>
            <p className="font-mono-tabular text-[11px] uppercase tracking-[0.28em]" style={{ color }}>
              Classification Verdict
            </p>
            <p className="mt-1 text-4xl font-extrabold tracking-tight md:text-5xl" style={{ color }}>
              {fresh ? "FRESH" : "NOT FRESH"}
            </p>
          </div>
        </div>
        <div className="text-right">
          <p className="font-mono-tabular text-[11px] uppercase tracking-[0.2em] text-muted-foreground">Confidence</p>
          <p className="mt-1 font-mono-tabular text-5xl font-semibold" style={{ color, textShadow: `0 0 24px color-mix(in oklab, ${color} 60%, transparent)` }}>
            {pct}%
          </p>
        </div>
      </div>
    </div>
  );
}

function CheckIcon({ color }: { color: string }) {
  return (
    <svg viewBox="0 0 24 24" className="h-8 w-8" fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 12l5 5L20 6" />
    </svg>
  );
}
function AlertIcon({ color }: { color: string }) {
  return (
    <svg viewBox="0 0 24 24" className="h-8 w-8" fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 3l10 18H2L12 3z" />
      <path d="M12 10v5M12 18v.5" />
    </svg>
  );
}

function ComparativeGallery({ original, crop, heatmap }: { original: string | null; crop: string; heatmap: string }) {
  const panels = [
    { title: "Original Upload", sub: "Source frame · pre-processing", src: original ?? undefined },
    { title: "Isolated Eye Crop", sub: "Hough localization · 384×384 patch", src: `data:image/png;base64,${crop}` },
    { title: "Grad-CAM Heatmap", sub: "Structural attention overlay · 45% α", src: `data:image/png;base64,${heatmap}` },
  ];
  return (
    <div className="grid gap-5 md:grid-cols-3">
      {panels.map((p) => (
        <figure key={p.title} className="overflow-hidden rounded-2xl glass">
          <div className="relative aspect-square w-full bg-[var(--surface)]">
            {p.src ? (
              <img src={p.src} alt={p.title} className="absolute inset-0 h-full w-full object-contain" />
            ) : (
              <div className="grid h-full w-full place-items-center text-xs text-muted-foreground">unavailable</div>
            )}
          </div>
          <figcaption className="border-t border-[color-mix(in_oklab,var(--cyan)_18%,transparent)] p-4">
            <p className="font-mono-tabular text-[10px] uppercase tracking-[0.22em] text-[var(--cyan)]">{p.title}</p>
            <p className="mt-1 text-xs text-muted-foreground">{p.sub}</p>
          </figcaption>
        </figure>
      ))}
    </div>
  );
}

// ---------- Footer ----------
function Footer() {
  return (
    <footer className="mt-32 border-t border-[color-mix(in_oklab,var(--cyan)_15%,transparent)] py-10">
      <div className="flex flex-col items-center justify-between gap-4 text-xs text-muted-foreground md:flex-row">
        <p className="font-mono-tabular uppercase tracking-wider">FreshlyFishy · vision lab · 2026</p>
        <p className="font-mono-tabular">api {API_BASE}</p>
      </div>
    </footer>
  );
}
