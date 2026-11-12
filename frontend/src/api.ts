// Thin typed wrapper around the FastAPI endpoints. Paths are relative (/api/...),
// which works through the Vite dev proxy and when FastAPI serves the built app.

import { useCallback, useEffect, useState } from "react";
import type { Benchmark, Metrics, ModelInfo, Prediction, Sample } from "./types";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError(0, "Cannot reach the API server. Start it with: uvicorn server.app:app --port 8000");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* body was not JSON */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  metrics: () => request<Metrics>("/api/metrics"),
  benchmark: () => request<Benchmark>("/api/benchmark"),
  model: () => request<ModelInfo>("/api/model"),
  samples: (n = 8) => request<Sample[]>(`/api/samples?n=${n}`),
  predict: (file: Blob) => {
    const body = new FormData();
    body.append("file", file, "image.png");
    return request<{ predictions: Prediction[] }>("/api/predict", { method: "POST", body });
  },
};

export interface Resource<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
  reload: () => void;
}

/** Fetch once on mount; expose loading / error state to the page. */
export function useResource<T>(loader: () => Promise<T>): Resource<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let live = true;
    setLoading(true);
    loader()
      .then((d) => live && (setData(d), setError(null)))
      .catch((e) => live && setError(e instanceof ApiError ? e : new ApiError(0, String(e))))
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick]);

  return { data, error, loading, reload: useCallback(() => setTick((t) => t + 1), []) };
}
