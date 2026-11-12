import { NavLink, Outlet } from "react-router-dom";
import type { ReactNode } from "react";

const icon = (d: string): ReactNode => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d={d} />
  </svg>
);

const links = [
  { to: "/", label: "Overview", end: true, icon: icon("M4 13h6V4H4zM14 20h6V4h-6zM4 20h6v-3H4z") },
  { to: "/training", label: "Training", end: false, icon: icon("M3 17l5-6 4 3 8-10M3 21h18") },
  { to: "/benchmark", label: "Benchmark", end: false, icon: icon("M12 14l4-4M5.6 19a9 9 0 1 1 12.8 0") },
  { to: "/playground", label: "Playground", end: false, icon: icon("M4 5h16v14H4zM4 15l4-4 4 4 3-3 5 5") },
];

/** App shell: fixed sidebar on desktop, top bar on mobile; pages render into <Outlet />. */
export default function Layout() {
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <svg width="16" height="16" viewBox="0 0 32 32"><path d="M14 7h5v12a6 6 0 0 1-6 6 6 6 0 0 1-5-2.6l3.2-2.6a2.2 2.2 0 0 0 1.8 1 1.8 1.8 0 0 0 1.9-1.9V7z" fill="#fff" /></svg>
          </div>
          <span>JAXVision</span>
        </div>
        <nav className="nav" aria-label="Main">
          {links.map((l) => (
            <NavLink key={l.to} to={l.to} end={l.end}>
              {l.icon}
              <span>{l.label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">
          <b>Built with</b>
          JAX, Flax, Optax, XLA
        </div>
      </aside>
      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
