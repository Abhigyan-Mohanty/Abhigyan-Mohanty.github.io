// Visitor Tracking
function logVisit() {
    const page = window.location.pathname.split("/").pop() || "index.html";
    const timestamp = new Date().toISOString();

    let visits = JSON.parse(localStorage.getItem('website_visits') || '[]');
    visits.push({ page, timestamp });

    // Keep only last 1000 logs to prevent storage bloat
    if (visits.length > 1000) visits = visits.slice(-1000);

    localStorage.setItem('website_visits', JSON.stringify(visits));
    console.log(`Visit logged: ${page} at ${timestamp}`);
}

// Scroll Motion Background Shift with Inertia
let lastScrollTop = window.pageYOffset || document.documentElement.scrollTop;
let targetIntensity = 0;
let currentIntensity = 0;
const smoothing = 0.1; // Lower = smoother/more inertia
const decay = 0.95;    // Rate at which intensity fades when stopped

function updateBackground() {
    // Smoothly interpolate current intensity towards target
    currentIntensity += (targetIntensity - currentIntensity) * smoothing;

    // Slowly decay target intensity so it returns to 0 when scrolling stops
    targetIntensity *= decay;
    if (targetIntensity < 0.001) targetIntensity = 0;

    // Apply color if there's enough intensity to notice
    if (currentIntensity > 0.001) {
        const bgColor = getComputedStyle(document.documentElement).getPropertyValue('--bg-color').trim();
        const motionColor = getComputedStyle(document.documentElement).getPropertyValue('--bg-motion-color').trim();
        document.body.style.backgroundColor = `color-mix(in srgb, ${motionColor} ${currentIntensity * 100}%, ${bgColor})`;
    } else if (document.body.style.backgroundColor !== '') {
        document.body.style.backgroundColor = '';
    }

    requestAnimationFrame(updateBackground);
}

// Start the animation loop
requestAnimationFrame(updateBackground);

window.addEventListener('scroll', () => {
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const distance = Math.abs(scrollTop - lastScrollTop);
    lastScrollTop = scrollTop;

    // Impact speed on target intensity
    const maxSpeed = 100;
    const speedFactor = Math.min(distance / maxSpeed, 1);

    // Add to target intensity (don't exceed 1)
    targetIntensity = Math.min(targetIntensity + speedFactor * 0.5, 1);
}, { passive: true });

// Initialize Feather icons and logging
document.addEventListener("DOMContentLoaded", () => {
    feather.replace();
    logVisit();
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
