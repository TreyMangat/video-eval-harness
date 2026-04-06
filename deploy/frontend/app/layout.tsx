import type { Metadata } from "next";
import { IBM_Plex_Mono, Space_Grotesk } from "next/font/google";

import "./globals.css";

const displayFont = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-display",
});

const monoFont = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  metadataBase: new URL("https://video-eval-harness-qu4m.vercel.app"),
  title: {
    default: "VBench — Multi-model video benchmark",
    template: "%s | VBench",
  },
  description:
    "Compare how 10 frontier vision-language models interpret the same video content. Agreement, accuracy, cost, and latency benchmarks.",
  openGraph: {
    title: "VBench — Multi-model video benchmark",
    description:
      "Compare how 10 frontier vision-language models interpret the same video content.",
    type: "website",
    siteName: "VBench",
  },
  twitter: {
    card: "summary_large_image",
    title: "VBench",
    description: "Multi-model video benchmark harness",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${displayFont.variable} ${monoFont.variable}`}>{children}</body>
    </html>
  );
}
