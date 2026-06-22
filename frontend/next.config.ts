import type { NextConfig } from "next";

const API = process.env.BACKEND_ORIGIN ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    // Django/DRF endpoints require a trailing slash (APPEND_SLASH). Next strips the
    // trailing slash when capturing `:path*`, so we re-append it on the destination.
    // Without this, Next forwards /api/foo to Django, Django 301s back to /api/foo/,
    // and the browser bounces between them forever (ERR_TOO_MANY_REDIRECTS).
    return [{ source: "/api/:path*", destination: `${API}/api/:path*/` }];
  },
};

export default nextConfig;
