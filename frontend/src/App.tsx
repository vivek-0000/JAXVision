import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Overview from "./pages/Overview";
import Training from "./pages/Training";
import Benchmark from "./pages/Benchmark";
import Playground from "./pages/Playground";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Overview />} />
        <Route path="training" element={<Training />} />
        <Route path="benchmark" element={<Benchmark />} />
        <Route path="playground" element={<Playground />} />
        <Route path="*" element={<Overview />} />
      </Route>
    </Routes>
  );
}
