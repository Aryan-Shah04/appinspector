import ghpages from 'gh-pages';

console.log('🚀 Starting deployment to GitHub Pages...');

const options = {
  branch: 'gh-pages',
  repo: undefined, // uses 'origin' by default
  // Setting history to false helps avoid "ENAMETOOLONG" errors on Windows
  // by preventing the library from cloning the entire git history into a deep cache folder.
  history: false, 
  dotfiles: true
};

ghpages.publish('dist', options, (err) => {
  if (err) {
    console.error('❌ Deployment Failed:', err);
    console.log('\n--- 🛠️ MANUAL FALLBACK INSTRUCTIONS ---');
    console.log('The automatic deploy script failed (this is common on Windows due to path limits).');
    console.log('Don\'t worry! You can deploy manually by running these 5 commands in your terminal:\n');
    console.log('  1. git checkout -b gh-pages');
    console.log('  2. npm run build');
    console.log('  3. git add dist -f');
    console.log('  4. git commit -m "Manual Deploy"');
    console.log('  5. git subtree push --prefix dist origin gh-pages');
    console.log('\nAfter step 5, your site will be live!');
    console.log('To go back to coding: git checkout main');
  } else {
    console.log('✅ Deployment Complete!');
    console.log('Check your repository settings > Pages to see your live site.');
  }
});