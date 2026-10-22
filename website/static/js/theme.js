// Inlined in both entry points before styles load, including the offline figure.
(() => {
  // Pages start light; a manual choice is remembered. The key changed when the default stopped
  // following the system, so choices saved under the old behaviour are dropped.
  const key = 'emg-tokenizer-theme-choice';
  let choice = null;
  try {
    const saved = localStorage.getItem(key);
    if (saved === 'light' || saved === 'dark') choice = saved;
  } catch { /* The page also works when browser storage is unavailable. */ }

  function apply() {
    const theme = choice || 'light';
    document.documentElement.dataset.theme = theme;
    const meta = document.querySelector('meta[name="theme-color"]');   // pages may set data-light/data-dark
    meta?.setAttribute('content', theme === 'dark' ? (meta.dataset.dark || '#151918') : (meta.dataset.light || '#f9f9f6'));
    document.querySelectorAll('[data-theme-toggle]').forEach(button => {
      button.setAttribute('aria-pressed', String(theme === 'dark'));
      button.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`);
      button.title = `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`;
    });
  }

  apply();
  document.addEventListener('DOMContentLoaded', apply);
  document.addEventListener('click', event => {
    if (!event.target.closest('[data-theme-toggle]')) return;
    choice = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    try { localStorage.setItem(key, choice); } catch { /* Keep the choice for this visit. */ }
    apply();
  });
  window.addEventListener('storage', event => {
    if (event.key !== key && event.key !== null) return;
    choice = ['light', 'dark'].includes(event.newValue) ? event.newValue : null;
    apply();
  });
})();
