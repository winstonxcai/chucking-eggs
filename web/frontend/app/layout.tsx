import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Guan Dan | 掼蛋",
  description: "Play the classic Chinese card game against AI opponents",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
