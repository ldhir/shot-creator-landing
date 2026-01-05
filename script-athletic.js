// Athletic Performance Landing Page Scripts

document.addEventListener('DOMContentLoaded', function() {
    // ========================================
    // NAVBAR SCROLL EFFECT
    // ========================================
    const navbar = document.querySelector('.navbar');
    let lastScroll = 0;

    window.addEventListener('scroll', () => {
        const currentScroll = window.scrollY;

        if (currentScroll > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }

        lastScroll = currentScroll;
    });

    // ========================================
    // INTERSECTION OBSERVER FOR ANIMATIONS
    // ========================================
    const observerOptions = {
        threshold: 0.15,
        rootMargin: '0px 0px -50px 0px'
    };

    const animationObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                // Once animated, stop observing
                animationObserver.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe all animatable elements
    const animatableElements = document.querySelectorAll(
        '.step-card, .benefit-card, .tweet-card'
    );

    animatableElements.forEach(el => {
        animationObserver.observe(el);
    });

    // ========================================
    // SMOOTH SCROLL FOR ANCHOR LINKS
    // ========================================
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                const navbarHeight = navbar.offsetHeight;
                const targetPosition = target.offsetTop - navbarHeight - 20;

                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });

    // ========================================
    // PARALLAX EFFECT FOR HERO
    // ========================================
    const hero = document.querySelector('.hero');
    const heroContent = document.querySelector('.hero-content');

    window.addEventListener('scroll', () => {
        if (window.scrollY < window.innerHeight) {
            const scrolled = window.scrollY;
            if (heroContent) {
                heroContent.style.transform = `translateY(${scrolled * 0.3}px)`;
                heroContent.style.opacity = 1 - (scrolled / (window.innerHeight * 0.8));
            }
        }
    });

    // ========================================
    // BUTTON HOVER EFFECTS
    // ========================================
    const buttons = document.querySelectorAll('.btn-glassy-primary, .btn-glassy, .btn-plan');

    buttons.forEach(button => {
        button.addEventListener('mouseenter', function(e) {
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            this.style.setProperty('--mouse-x', `${x}px`);
            this.style.setProperty('--mouse-y', `${y}px`);
        });
    });

    // ========================================
    // CARD TILT EFFECT
    // ========================================
    const cards = document.querySelectorAll('.step-card, .benefit-card');

    cards.forEach(card => {
        card.addEventListener('mousemove', function(e) {
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const centerX = rect.width / 2;
            const centerY = rect.height / 2;

            const rotateX = (y - centerY) / 20;
            const rotateY = (centerX - x) / 20;

            this.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-8px)`;
        });

        card.addEventListener('mouseleave', function() {
            this.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) translateY(0)';
        });
    });

    // ========================================
    // SECTION TITLES ANIMATION
    // ========================================
    const sectionTitles = document.querySelectorAll('.section-title');

    const titleObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
                titleObserver.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.5
    });

    sectionTitles.forEach(title => {
        title.style.opacity = '0';
        title.style.transform = 'translateY(30px)';
        title.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        titleObserver.observe(title);
    });

    // ========================================
    // VIDEO CONTAINER GLOW ON HOVER
    // ========================================
    const videoContainer = document.querySelector('.video-container');

    if (videoContainer) {
        videoContainer.addEventListener('mouseenter', function() {
            this.style.boxShadow = '0 0 80px rgba(255, 77, 0, 0.3)';
        });

        videoContainer.addEventListener('mouseleave', function() {
            this.style.boxShadow = '0 40px 100px rgba(0, 0, 0, 0.6)';
        });
    }

    // ========================================
    // COUNTER ANIMATION FOR STATS (if added)
    // ========================================
    function animateCounter(element, target, duration = 2000) {
        let start = 0;
        const increment = target / (duration / 16);
        const timer = setInterval(() => {
            start += increment;
            if (start >= target) {
                element.textContent = target.toLocaleString();
                clearInterval(timer);
            } else {
                element.textContent = Math.floor(start).toLocaleString();
            }
        }, 16);
    }

    // ========================================
    // KEYBOARD NAVIGATION SUPPORT
    // ========================================
    document.addEventListener('keydown', (e) => {
        // Press 'Enter' on focused buttons
        if (e.key === 'Enter' && document.activeElement.classList.contains('btn-plan')) {
            document.activeElement.click();
        }
    });

    // ========================================
    // LOADING COMPLETE - REVEAL PAGE
    // ========================================
    document.body.style.opacity = '1';

    console.log('ShotSync Athletic Theme Loaded');
});

// ========================================
// CURSOR TRAIL EFFECT (Optional - Performance Heavy)
// ========================================
/*
const cursor = document.createElement('div');
cursor.className = 'custom-cursor';
document.body.appendChild(cursor);

document.addEventListener('mousemove', (e) => {
    cursor.style.left = e.clientX + 'px';
    cursor.style.top = e.clientY + 'px';
});
*/
