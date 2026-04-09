/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brandPrimary: '#4f46e5', // The purple from your mockups
        brandDark: '#111827',    // The dark sidebar color
      }
    },
  },
  plugins: [],
}