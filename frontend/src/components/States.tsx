import type { ReactNode } from "react";
import type { Resource } from "../api";

export function PageHead({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <header className="page-head">
      <h1>{title}</h1>
      {children && <p>{children}</p>}
    </header>
  );
}

export function Loading() {
  return <div className="skeleton" role="status" aria-label="Loading" />;
}

/** Shown when the API answered 404 (nothing trained yet) or could not be reached at all. */
export function Unavailable({ res, command }: { res: Resource<unknown>; command: string }) {
  const offline = res.error?.status === 0;
  return (
    <section className="panel state">
      <h2>{offline ? "The API server is not running" : "No results yet"}</h2>
      <p>
        {offline
          ? "The dashboard could not reach the Python backend. Start it from the project root:"
          : "Train a model and this page fills in automatically. From the project root, run:"}
      </p>
      <pre>{offline ? "uvicorn server.app:app --port 8000" : command}</pre>
      <p style={{ marginTop: 14 }}>
        <button className="btn" onClick={res.reload}>Check again</button>
      </p>
    </section>
  );
}

/** Wraps a resource: loading skeleton, empty/error state, or the children once data exists. */
export function Gate<T>({
  res,
  command,
  children,
}: {
  res: Resource<T>;
  command: string;
  children: (data: T) => ReactNode;
}) {
  if (res.loading && !res.data) return <Loading />;
  if (res.error && !res.data) {
    return res.error.status === 0 || res.error.status === 404 ? (
      <Unavailable res={res} command={command} />
    ) : (
      <section className="panel state">
        <h2>Something went wrong</h2>
        <p>{res.error.message}</p>
      </section>
    );
  }
  return <>{res.data && children(res.data)}</>;
}
