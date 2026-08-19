import type { Metadata } from "next";
import { Plus_Jakarta_Sans, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const jakarta = Plus_Jakarta_Sans({
  subsets: ["latin"],
  variable: "--font-jakarta",
});

const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "DocuMind AI — Grounded Intelligence Engine",
  description: "Enterprise RAG system with 768-dim pgvector retrieval and inline provenance citations.",
  icons: {
    icon: "/icon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${jakarta.variable} ${mono.variable} bg-[#070A12] text-slate-100 antialiased selection:bg-emerald-500/20 selection:text-emerald-300`}>
        {children}
      </body>
    </html>
  );
}