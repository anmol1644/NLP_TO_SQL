/** @type {import('next').NextConfig} */
const nextConfig = {
  // Use a different output directory to avoid potential symlink issues
  distDir: 'build',
  // Keep any other existing settings
  reactStrictMode: true
};

module.exports = nextConfig; 