/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
      "./index.html",
      "./src/**/*.{js,ts,jsx,tsx}" // scans all your source files
    ],
    theme: {
      extend: {
        fontFamily: {
          // Body / general text
          sans: ['Inter', 'sans-serif'],
          // Headings / brand
          brand: ['Archivo Black', 'sans-serif'],
        },
      },
    },
    plugins: [],
  };
  