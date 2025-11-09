// WebGL Liquid Glass Background
class WebGLBackground {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = canvas.getContext('webgl2');
        this.program = null;
        this.scale = Math.max(1, 0.5 * window.devicePixelRatio);
        this.mouseX = 0;
        this.mouseY = 0;
        this.time = 0;

        this.vertexShader = `#version 300 es
            precision highp float;
            in vec4 position;
            void main() {
                gl_Position = position;
            }`;

        this.fragmentShader = `#version 300 es
            precision highp float;
            out vec4 fragColor;
            uniform vec2 resolution;
            uniform float time;
            uniform vec2 mouse;

            #define S(a, b, t) smoothstep(a, b, t)
            #define PI 3.14159265359

            // Noise functions
            vec3 hash3(vec2 p) {
                vec3 q = vec3(dot(p, vec2(127.1, 311.7)),
                             dot(p, vec2(269.5, 183.3)),
                             dot(p, vec2(419.2, 371.9)));
                return fract(sin(q) * 43758.5453);
            }

            float noise(vec2 p) {
                vec2 i = floor(p);
                vec2 f = fract(p);
                vec2 u = f * f * (3.0 - 2.0 * f);

                float a = dot(hash3(i + vec2(0.0, 0.0)).xy, f - vec2(0.0, 0.0));
                float b = dot(hash3(i + vec2(1.0, 0.0)).xy, f - vec2(1.0, 0.0));
                float c = dot(hash3(i + vec2(0.0, 1.0)).xy, f - vec2(0.0, 1.0));
                float d = dot(hash3(i + vec2(1.0, 1.0)).xy, f - vec2(1.0, 1.0));

                return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
            }

            float fbm(vec2 p) {
                float value = 0.0;
                float amplitude = 0.5;
                float frequency = 1.0;

                for(int i = 0; i < 6; i++) {
                    value += amplitude * noise(p * frequency);
                    frequency *= 2.0;
                    amplitude *= 0.5;
                }
                return value;
            }

            // Liquid glass distortion
            vec2 liquidDistortion(vec2 uv, float t) {
                vec2 p = uv * 3.0;

                // Multiple layers of flowing noise
                float n1 = fbm(p + t * 0.3 + vec2(fbm(p + t * 0.2), fbm(p - t * 0.15)));
                float n2 = fbm(p - t * 0.2 + vec2(fbm(p - t * 0.3), fbm(p + t * 0.25)));

                vec2 distortion = vec2(n1, n2) * 0.03;

                // Mouse interaction
                vec2 mouseInfluence = (uv - mouse) * 2.0;
                float mouseDist = length(mouseInfluence);
                distortion += mouseInfluence * (1.0 / (1.0 + mouseDist * 3.0)) * 0.02;

                return uv + distortion;
            }

            // Simple flowing gradient effect
            vec3 simpleGradient(vec2 uv, float t) {
                // Gentle flowing pattern
                float pattern = fbm(uv * 1.5 + t * 0.1);

                // Light color palette (whites to soft peach/coral)
                vec3 col1 = vec3(1.0, 0.98, 0.96);     // Soft white
                vec3 col2 = vec3(1.0, 0.94, 0.92);     // Pale peach
                vec3 col3 = vec3(1.0, 0.90, 0.88);     // Light coral

                // Smooth color transitions
                vec3 color = mix(col1, col2, S(0.3, 0.6, pattern));
                color = mix(color, col3, S(0.5, 0.8, pattern) * 0.3);

                return color;
            }

            void main() {
                vec2 uv = gl_FragCoord.xy / resolution;
                vec2 centeredUV = (gl_FragCoord.xy - 0.5 * resolution) / min(resolution.x, resolution.y);

                float t = time * 0.5;

                // Get base color with gentle flow
                vec3 color = simpleGradient(centeredUV, t);

                // Add subtle top-to-bottom gradient
                float gradientY = uv.y;
                color = mix(color, vec3(1.0, 0.96, 0.94), gradientY * 0.2);

                fragColor = vec4(color, 1.0);
            }`;

        this.init();
    }

    init() {
        const gl = this.gl;

        // Create shaders
        const vs = gl.createShader(gl.VERTEX_SHADER);
        const fs = gl.createShader(gl.FRAGMENT_SHADER);

        gl.shaderSource(vs, this.vertexShader);
        gl.shaderSource(fs, this.fragmentShader);

        gl.compileShader(vs);
        gl.compileShader(fs);

        // Create program
        this.program = gl.createProgram();
        gl.attachShader(this.program, vs);
        gl.attachShader(this.program, fs);
        gl.linkProgram(this.program);

        // Create buffer
        const buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, 1, -1, -1, 1, 1, 1, -1]), gl.STATIC_DRAW);

        const position = gl.getAttribLocation(this.program, 'position');
        gl.enableVertexAttribArray(position);
        gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);

        this.resize();
        this.setupMouseTracking();
        this.render();
    }

    resize() {
        const dpr = this.scale;
        this.canvas.width = window.innerWidth * dpr;
        this.canvas.height = window.innerHeight * dpr;
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    }

    setupMouseTracking() {
        this.canvas.addEventListener('mousemove', (e) => {
            this.mouseX = e.clientX / window.innerWidth;
            this.mouseY = 1.0 - (e.clientY / window.innerHeight);
        });
    }

    render() {
        const gl = this.gl;

        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.useProgram(this.program);

        const resolutionLoc = gl.getUniformLocation(this.program, 'resolution');
        const timeLoc = gl.getUniformLocation(this.program, 'time');
        const mouseLoc = gl.getUniformLocation(this.program, 'mouse');

        gl.uniform2f(resolutionLoc, this.canvas.width, this.canvas.height);
        gl.uniform1f(timeLoc, this.time);
        gl.uniform2f(mouseLoc, this.mouseX, this.mouseY);

        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

        this.time += 0.016;
        requestAnimationFrame(() => this.render());
    }
}

// Initialize WebGL backgrounds on all sections
document.addEventListener('DOMContentLoaded', function() {
    // Hero canvas
    const heroCanvas = document.getElementById('hero-canvas');
    if (heroCanvas) {
        const heroBg = new WebGLBackground(heroCanvas);
        window.addEventListener('resize', () => heroBg.resize());
    }

    // Section canvases
    const sectionCanvases = document.querySelectorAll('.section-canvas');
    const backgrounds = [];

    sectionCanvases.forEach((canvas, index) => {
        // Add slight time offset for each section for variety
        const bg = new WebGLBackground(canvas);
        bg.time = index * 10; // Offset time for variation
        backgrounds.push(bg);
    });

    window.addEventListener('resize', () => {
        backgrounds.forEach(bg => bg.resize());
    });

    // FAQ Accordion functionality
    const faqItems = document.querySelectorAll('.faq-item');

    faqItems.forEach(item => {
        const question = item.querySelector('.faq-question');

        question.addEventListener('click', () => {
            // Close other open items
            faqItems.forEach(otherItem => {
                if (otherItem !== item && otherItem.classList.contains('active')) {
                    otherItem.classList.remove('active');
                }
            });

            // Toggle current item
            item.classList.toggle('active');
        });
    });
});

// Smooth scroll for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add scroll animation for sections
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -100px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe all major sections
document.addEventListener('DOMContentLoaded', () => {
    const sections = document.querySelectorAll('.benefit-card, .step-card, .pricing-card, .testimonial-card');

    sections.forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(30px)';
        section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(section);
    });
});

// Navbar background on scroll
window.addEventListener('scroll', () => {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 50) {
        navbar.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.1)';
    } else {
        navbar.style.boxShadow = '0 1px 3px rgba(0, 0, 0, 0.05)';
    }
});

// Newsletter form submission
document.querySelector('.newsletter-form')?.addEventListener('submit', function(e) {
    e.preventDefault();
    const email = this.querySelector('input[type="email"]').value;
    if (email) {
        alert('Thank you for subscribing! We\'ll keep you updated.');
        this.querySelector('input[type="email"]').value = '';
    }
});

// Get Started button actions
document.querySelectorAll('.btn-primary, .btn-cta, .btn-plan, .btn-hero').forEach(button => {
    button.addEventListener('click', function(e) {
        if (!this.getAttribute('href')) {
            e.preventDefault();
            // Replace with actual signup/demo logic
            console.log('Button clicked:', this.textContent);
            alert('Welcome to ShotSync! This would open the signup/demo form.');
        }
    });
});

// Add hover effect to pricing cards
document.querySelectorAll('.pricing-card').forEach(card => {
    card.addEventListener('mouseenter', function() {
        this.style.zIndex = '10';
    });

    card.addEventListener('mouseleave', function() {
        this.style.zIndex = '1';
    });
});

// Mobile menu toggle (for future mobile menu implementation)
const createMobileMenu = () => {
    const navbar = document.querySelector('.navbar');
    const navLinks = document.querySelector('.nav-links');

    if (window.innerWidth <= 768) {
        // Mobile menu functionality can be added here
        console.log('Mobile view detected');
    }
};

window.addEventListener('resize', createMobileMenu);
createMobileMenu();
