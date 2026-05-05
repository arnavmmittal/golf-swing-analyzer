import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Golf Swing Analyzer",
  description:
    "Upload your swing, get specific biomechanics-grounded feedback toward a tour-level swing.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#faf8f3] text-fairway-900 antialiased">
        {children}
      </body>
    </html>
  );
}
