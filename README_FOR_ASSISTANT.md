# Roots Landing Page - Documentation Index

**For:** Assistant reviewing the Roots/ShotSync landing page design
**Created:** November 2025

---

## 📚 Which Document Should I Read?

### For Quick Overview (5 minutes)
**→ Read:** `QUICK_REFERENCE.md`
- Colors, fonts, and key effects
- Page structure overview
- 8 sections summarized
- Design principles
- Brand voice

### For Visual Understanding (10 minutes)
**→ Read:** `SECTION_BREAKDOWN.md`
- Section-by-section visual diagrams
- Exact specifications with ASCII layouts
- Hover states and interactions
- Mobile responsive changes
- Implementation checklist

### For Complete Deep Dive (30 minutes)
**→ Read:** `DESIGN_DOCUMENTATION.md`
- Comprehensive design system
- Detailed section breakdowns
- Technical implementation
- Content strategy
- SEO, analytics, performance
- Future enhancements

---

## 🎯 Quick Start

If you need to understand the design RIGHT NOW:

### 1. Look at the live page
```
Open: index.html in a browser
```

### 2. Read this summary:

**What it is:**
- Landing page for "ShotSync by Roots AI"
- AI shot analysis tool for athletes

**Visual style:**
- Coral pink (#FF6B7A) brand color
- White → pink gradient background (every section)
- Glassmorphic (blur) effects on buttons
- Forum serif + Courier Prime monospace fonts

**Structure (9 sections):**
1. Navbar - Sticky, links to sections
2. Hero - Full screen, "Try ShotSync" CTA
3. How It Works - 3 steps in dark cards
4. Demo Video - ShotSync in action
5. Features - Problem → Solution → Vision (3 cards)
6. Testimonials - Twitter wall with #rooted
7. Feedback - Embedded Typeform
8. Final CTA - Gradient coral box, "Get Started Free"
9. Footer - Dark background, links

**Key design elements:**
- All buttons lift on hover (glassmorphic style)
- All cards are dark with light text
- Multiple CTAs → `./tool/index.html` (ShotSync tool)
- Mobile responsive at 768px breakpoint

### 3. If you need more detail, choose a doc above

---

## 📁 File Structure

```
shot-creator-landing/
├── index.html                        ← Main landing page
├── styles.css                        ← All styles (~1000 lines)
├── script.js                         ← Interactivity
├── twitter-wall.js                   ← Twitter integration
├── RootsLogo.png                     ← Brand logo
│
├── README_FOR_ASSISTANT.md          ← This file (start here!)
├── QUICK_REFERENCE.md                ← 5-min overview
├── SECTION_BREAKDOWN.md              ← Visual diagrams
├── DESIGN_DOCUMENTATION.md           ← Complete specs
│
└── tool/
    └── index.html                    ← ShotSync tool (linked from landing)
```

---

## 🎨 Design System at a Glance

### Colors
```
Primary:    #FF6B7A  ← Coral pink (brand color)
Hover:      #FF5468  ← Darker coral
Dark:       #1a1a1a  ← Text
Gray:       #666666  ← Secondary text
Cards:      hsl(222 47% 11%)  ← Dark blue-gray

Gradient Background (all sections):
linear-gradient(90deg,
  white 0%,
  rgba(255, 179, 186, 1) 85%,
  rgba(255, 107, 122, 1) 100%
)
```

### Typography
```
Body/Headings:  Forum (serif) - elegant
UI/Buttons:     Courier Prime (monospace) - technical

Sizes:
- Hero: 64px (40px mobile)
- Sections: 48px (32px mobile)
- Body: 16-18px
```

### Effects
```
Glassmorphism:
- background: rgba(255, 255, 255, 0.1)
- backdrop-filter: blur(12px)
- border: 1px solid rgba(255, 255, 255, 0.2)

Hover:
- transform: translateY(-2px) for buttons
- transform: translateY(-4px) for cards
- box-shadow increases

Transitions:
- all 0.3s ease
```

### Spacing
```
Sections:    100px padding (vertical)
Container:   1200px max-width
Cards:       16px border-radius
Buttons:     30px border-radius (pill shape)
```

---

## 🔗 Important Links

**Main CTA (appears 4 times):**
```
./tool/index.html  ← ShotSync tool
```

**Social:**
```
https://x.com/with__roots  ← Twitter (active link)
```

**Embeds:**
```
Typeform: https://form.typeform.com/to/wUkR5vhP
Icons:    https://img.icons8.com/fluency/96/[icon].png
```

---

## 🎯 Key Sections Explained

### 1. Hero
- Full viewport height
- Diagonal grid pattern background
- Noise texture overlay
- Large centered text
- Glassy coral CTA button

### 2. How It Works
- 3 dark cards in a row
- Numbered circles (1, 2, 3)
- Simple 3-step explanation

### 3. Features (Most Important!)
```
[PROBLEM]          [SOLUTION]         [VISION]
Recruiting         ShotSync           Full AI
is broken          (LIVE NOW)         Platform
                   ← highlighted!
                   ← has CTA
```
Middle card is the key - highlighted with 2px coral border

### 4. Testimonials
- Grid of tweets with #rooted
- Social proof
- Twitter share button

---

## 💼 Brand Guidelines

**Voice:**
- Empowering: "Can you shoot like a pro?"
- Inclusive: "Every Athlete Has Roots"
- Direct: "The recruiting game is broken. We're fixing it."

**Positioning:**
- Problem: College recruiting is expensive, fragmented, broken
- Solution: ShotSync (AI shot analysis) - live now, free trial
- Vision: Complete AI platform for athletes (coming soon)

**Target Audience:**
- Athletes (basketball)
- Coaches
- Parents of young athletes

---

## 📱 Responsive Design

**Breakpoint:** 768px

**Mobile changes:**
- Hide nav links (show logo + badge only)
- All grids → single column
- Text 30% smaller
- Padding reduced
- Footer stacks vertically

---

## ✅ Quick Implementation Checklist

If implementing/modifying, ensure:

**Visual:**
- [ ] Coral pink (#FF6B7A) used consistently
- [ ] Gradient background on all sections
- [ ] Glassmorphism on buttons
- [ ] Dark cards with light text
- [ ] Lift on hover (2-4px)

**Structure:**
- [ ] 9 sections total
- [ ] Sticky navbar
- [ ] Multiple CTAs to ./tool/index.html
- [ ] Dark footer

**Typography:**
- [ ] Forum for content
- [ ] Courier Prime for UI
- [ ] Consistent size scale

**Responsive:**
- [ ] Mobile breakpoint at 768px
- [ ] Grids stack to single column
- [ ] Text sizes reduce

---

## 🤔 Common Questions

**Q: What is ShotSync?**
A: AI tool that compares athlete's basketball shooting form to professional players

**Q: What is Roots?**
A: Full AI platform for athletes (recruiting, coaching, highlights, etc.). ShotSync is the first product.

**Q: Why coral pink?**
A: Warm, energetic, differentiates from typical sports blue. Approachable but premium.

**Q: Why glassmorphism?**
A: Modern, tech-forward aesthetic that matches AI positioning

**Q: Why so many CTAs?**
A: Free trial = low friction. Multiple touchpoints = higher conversion.

**Q: What's the goal?**
A: Get athletes to try ShotSync tool (free trial)

---

## 📊 Design Principles

### 1. Problem → Solution → Vision
- Show the broken system
- Introduce working solution (ShotSync)
- Tease complete platform

### 2. Multiple Conversion Points
Every section has a path to the product:
- Hero CTA
- Features card CTA
- Final CTA
- Navbar badge

### 3. Social Proof
Twitter wall shows real users sharing results

### 4. Modern + Approachable
- Glassmorphism = cutting-edge tech
- Coral pink = warm, friendly
- Mix of serif + mono = professional + precise

---

## 🚀 Next Steps for Your Assistant

### To Understand Design:
1. Open `index.html` in browser
2. Read `QUICK_REFERENCE.md` (5 min)
3. Scan `SECTION_BREAKDOWN.md` for visuals
4. Deep dive in `DESIGN_DOCUMENTATION.md` if needed

### To Modify:
1. Check which section needs changes
2. Find it in `SECTION_BREAKDOWN.md` for specs
3. Update in `index.html` + `styles.css`
4. Test responsive at 768px

### To Implement From Scratch:
1. Start with `DESIGN_DOCUMENTATION.md`
2. Use color/typography system
3. Follow section structure
4. Copy spacing/effects

---

## 📞 Need Help?

**For specific questions:**
- Colors/fonts → See `QUICK_REFERENCE.md` → Design System
- Layout/spacing → See `SECTION_BREAKDOWN.md` → Section diagrams
- Implementation → See `DESIGN_DOCUMENTATION.md` → Technical Implementation

**Can't find something?**
- Search in `DESIGN_DOCUMENTATION.md` (most comprehensive)
- Look at actual code in `index.html` + `styles.css`

---

## Summary

**In one sentence:**
Modern landing page with coral pink brand color, glassmorphic effects, and a problem→solution→vision narrative to convert athletes to try the free ShotSync tool.

**Key files to review:**
1. `index.html` - The actual page
2. `QUICK_REFERENCE.md` - Fast overview
3. `SECTION_BREAKDOWN.md` - Visual specs
4. `DESIGN_DOCUMENTATION.md` - Everything

**Happy reviewing! 🏀**
