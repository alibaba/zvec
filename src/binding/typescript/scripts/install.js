// Install script for @zvec/zvec
// Tries to use pre-built native addon if available for this platform.
// Falls back to node-gyp rebuild if no prebuild exists.

const path = require('path');
const fs = require('fs');

const platform = process.platform;
const arch = process.arch;
const prebuildDir = path.join(__dirname, '..', 'prebuilds', `${platform}-${arch}`);
const addonPath = path.join(prebuildDir, 'zvec_addon.node');

if (fs.existsSync(addonPath)) {
  // Pre-built binary available — copy to build/Release for require() to find
  const releaseDir = path.join(__dirname, '..', 'build', 'Release');
  fs.mkdirSync(releaseDir, { recursive: true });
  fs.copyFileSync(addonPath, path.join(releaseDir, 'zvec_addon.node'));
  console.log(`@zvec/zvec: using pre-built native addon for ${platform}-${arch}`);
} else {
  console.log(`@zvec/zvec: no prebuild for ${platform}-${arch}, falling back to node-gyp`);
  // Exiting with error triggers the `|| node-gyp rebuild` fallback in package.json
  process.exit(1);
}
