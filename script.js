// Initialize Feather icons
document.addEventListener("DOMContentLoaded", () => {
    feather.replace();
});

// Simple page transiton animation helper
window.addEventListener('pageshow', (event) => {
    if (event.persisted) {
        // Clear animation cache if loaded from bfcache
        document.body.classList.remove('animate-in');
        void document.body.offsetWidth; // trigger reflow
        document.body.classList.add('animate-in');
    }
});
