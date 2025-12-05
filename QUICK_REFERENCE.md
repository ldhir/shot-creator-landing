# Roots Landing Page - Quick Reference Guide

**For:** Your assistant to quickly understand the design
**Read time:** 5 minutes

---

## What Is It?

**Product:** ShotSync by Roots AI
**Purpose:** AI shot analysis tool - athletes compare their basketball shooting form to pros
**Tagline:** "Cause Every Athlete Has Roots!"

---

## Visual Style

### Colors
- **Brand Color:** Coral pink `#FF6B7A`
- **Background:** White fading to pink gradient (used on every section)
- **Cards:** Dark blue-gray `hsl(222 47% 11%)`
- **Text:** Dark `#1a1a1a` and gray `#666666`

### Fonts
- **Headings/Body:** Forum (serif) - elegant
- **Buttons/Nav:** Courier Prime (monospace) - technical

### Key Visual Effects
1. **Glassmorphism** - Semi-transparent blur effect on buttons
2. **Noise Texture** - Subtle grain on hero background
3. **Diagonal Grid** - Brand pattern in hero section
4. **Gradient** - White → Light pink → Coral on all sections

---

## Page Structure (8 Sections)

### 1. Navbar (Sticky)
```
[Logo] [Get Demo | About | Questions] [ShotSync Badge]
```
- Stays at top while scrolling
- Semi-transparent with blur
- Links scroll to sections

### 2. Hero
**What it says:**
- "Roots - An all in one AI athlete assistant!"
- "We connect athletes, coaches, and content creators..."
- Button: "Try ShotSync"

**Design:**
- Full screen height
- Centered text
- Diagonal grid pattern background
- Grain texture overlay

### 3. How It Works (3 Steps)
```
[1. Choose Player] [2. Record Shot] [3. Get Results]
```
- 3 dark cards in a row
- Numbered circles (coral background)
- Simple explanation

### 4. Demo Video
- Title: "See ShotSync in Action"
- Subtitle: "Train like a pro without the pro price tag"
- Embedded video player (16:9 ratio)

### 5. Features (3 Cards)

```
[THE PROBLEM]         [LIVE NOW ★]           [THE VISION]
College recruiting    ShotSync - AI Shot     One AI Platform
is broken             Analysis               for Everything
• Stats               • Compare to pros      • AI Coach (JARVIS)
• Cost issues         • Real-time feedback   • Auto Highlights
• etc.                • Track progress       • Smart Recruiting
                      [Try Free Now →]       • etc.
```

**Middle card highlighted:**
- 2px coral border
- Most prominent
- Has CTA button

### 6. Testimonials (Twitter Wall)
- Grid of tweets with #rooted hashtag
- Shows user similarity scores
- "Share Your Score on Twitter" button

### 7. Feedback Form
- "We'd Love Your Feedback"
- Embedded Typeform
- 600px height, centered

### 8. Final CTA
- Gradient coral box
- "Can you shoot like a pro?"
- "Get Started Free" button

### 9. Footer
```
[Logo/Copyright] [Privacy|Terms|Contact] [Twitter|LinkedIn|Instagram]
```
- Dark background
- 3 sections horizontal
- Stacks vertically on mobile

---

## Responsive Design

**Breakpoint:** 768px

**Mobile changes:**
- Nav links hidden (logo + badge remain)
- All grids → single column
- Text sizes reduced:
  - Hero: 64px → 40px
  - Sections: 48px → 32px
- Footer stacks and centers

---

## Interactive Elements

### Buttons (4 types)

1. **Glassy Coral** (Hero, CTAs)
   - Semi-transparent coral
   - Blur effect
   - Rounded pill shape

2. **Solid Coral** (Features card)
   - "Try Free Now →"
   - White text on coral

3. **Glassy White** (Final CTA)
   - Semi-transparent white
   - For coral background

4. **Twitter Blue**
   - "Share Your Score"
   - Twitter brand color

**All buttons:**
- Lift 2px on hover
- Add shadow
- Smooth transitions

### Cards
- All have 16px rounded corners
- Lift 4px on hover
- Dark background
- White/light text

---

## Key Links

**Internal:**
- `./tool/index.html` - ShotSync tool (main product)

**External:**
- Twitter: `https://x.com/with__roots`
- Typeform: `https://form.typeform.com/to/wUkR5vhP`
- Icons: Icons8 CDN

**Section Anchors:**
- `#demo` - Video section
- `#benefits` - Features section
- `#faq` - Feedback form

---

## File Structure

```
shot-creator-landing/
├── index.html              ← Main page
├── styles.css              ← All styles
├── script.js               ← Interactivity
├── twitter-wall.js         ← Twitter integration
├── RootsLogo.png           ← Logo
└── tool/
    └── index.html          ← ShotSync tool
```

---

## Design Principles

### 1. Problem → Solution → Vision
- **Problem:** Recruiting is broken
- **Solution:** ShotSync (live now)
- **Vision:** Complete AI platform

### 2. Multiple CTAs
Every section has a way to try the product:
- Hero
- Features card
- Final CTA
- Navbar badge

### 3. Social Proof
Twitter wall shows real users sharing scores

### 4. Modern + Approachable
- Glassmorphism = modern/tech
- Coral pink = warm/friendly
- Serif + mono fonts = professional + technical

---

## Content Strategy

### Headlines Progression
1. "Roots - An all in one AI athlete assistant"
2. "See ShotSync in Action"
3. "Features" (Problem/Solution/Vision)
4. "See What Athletes Are Sharing"
5. "We'd Love Your Feedback"
6. "Can you shoot like a pro?"

### CTAs Progression
1. "Try ShotSync" - exploratory
2. "Try Free Now →" - directional
3. "Get Started Free" - commitment

---

## Brand Voice

**Tone:** Empowering but friendly
- "Can you shoot like a pro?" (challenge)
- "Every Athlete Has Roots" (inclusive)
- "We're fixing it" (direct, confident)

**Language:**
- Simple, clear
- Action-oriented
- Problem-aware
- Solution-focused

---

## Technical Notes

**Fonts:** Google Fonts (preconnected)
**Layout:** CSS Grid + Flexbox
**Effects:** Backdrop-filter, gradients, transforms
**Video:** HTML5 with fallbacks
**Forms:** Typeform iframe
**Icons:** Icons8 CDN

**Mobile-first:** Responsive breakpoint at 768px

---

## Quick Implementation Checklist

If implementing/modifying:

**Colors:**
- [ ] Use `#FF6B7A` for all brand touches
- [ ] Use gradient background on all sections
- [ ] Use dark cards for content

**Spacing:**
- [ ] 100px section padding
- [ ] 1200px max container width
- [ ] 16px card border-radius

**Typography:**
- [ ] Forum for text
- [ ] Courier Prime for UI elements
- [ ] 64px hero, 48px sections

**Effects:**
- [ ] Glassmorphism on buttons
- [ ] Lift on hover (2-4px)
- [ ] Smooth transitions (0.3s)

---

## Common Questions

**Q: Why glassmorphism?**
A: Modern, premium feel that matches AI/tech positioning

**Q: Why coral pink?**
A: Warm, energetic, differentiates from typical sports brand blue

**Q: Why mix serif + mono fonts?**
A: Elegance (Forum) + technical precision (Courier Prime)

**Q: Why so many CTAs?**
A: Multiple touchpoints = higher conversion. Free trial = low friction.

**Q: Why dark cards?**
A: Contrast with light backgrounds, emphasize content, modern aesthetic

---

## Modification Tips

### To change brand color:
1. Update `--primary-color` and `--primary-hover` in `:root`
2. Gradient will automatically update

### To add a section:
1. Copy existing section HTML
2. Update content
3. Maintain 100px padding
4. Use same gradient background

### To modify responsive:
1. Edit `@media (max-width: 768px)` in styles.css
2. Test grid → column behavior
3. Check text size reductions

---

## Summary

**In one sentence:**
A modern, coral-pink landing page that presents Roots' AI shot analysis tool (ShotSync) through a clear problem→solution→vision narrative with multiple conversion points and social proof.

**Key takeaways:**
- Coral pink (#FF6B7A) is the brand
- Glassmorphism = modern tech feel
- Problem-solution-vision structure
- Multiple low-friction CTAs ("Try Free")
- Social proof through Twitter integration
- Mobile-responsive single-page design

---

**Need more detail?** See `DESIGN_DOCUMENTATION.md` for comprehensive breakdown.
