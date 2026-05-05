import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        fairway: {
          50: "#f3f7f0",
          100: "#e3ecdc",
          500: "#5a8a3a",
          600: "#46702a",
          700: "#365720",
          900: "#1d2f10",
        },
        sand: {
          50: "#faf6ee",
          100: "#f3eada",
          500: "#c8a96a",
        },
      },
      fontFamily: {
        display: ["ui-serif", "Georgia", "Cambria", "serif"],
      },
    },
  },
  plugins: [],
};

export default config;
