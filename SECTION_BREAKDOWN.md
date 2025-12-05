# Roots Landing Page - Visual Section Breakdown

**For your assistant:** Section-by-section visual guide with exact specifications

---

## 🎨 Color Swatches

```
Coral Pink:     ███ #FF6B7A (primary brand color)
Coral Hover:    ███ #FF5468 (hover states)
Dark Text:      ███ #1a1a1a (headings, body)
Gray Text:      ███ #666666 (subheadings, secondary)
Dark Cards:     ███ hsl(222 47% 11%) (card backgrounds)
White:          ███ #FFFFFF
```

**Gradient (used on EVERY section background):**
```
Left → Right:
White (0%) → Light Pink (85%) → Coral (100%)
```

---

## 📐 Section 1: NAVBAR

```
┌────────────────────────────────────────────────────────────┐
│  🏀 Roots Logo    Get Demo  About  Questions     ShotSync  │ ← Sticky, stays at top
└────────────────────────────────────────────────────────────┘
```

**Specs:**
- Height: 90px (20px padding top/bottom)
- Background: Semi-transparent white (`rgba(255, 255, 255, 0.8)`)
- Effect: Glassmorphic blur (`backdrop-filter: blur(10px)`)
- Border bottom: 1px solid light pink

**Logo:**
- Height: 50px
- File: `RootsLogo.png`

**Nav Links:**
- Font: Courier Prime (monospace)
- Size: 15px
- Color: Dark on rest, coral on hover
- Gap: 40px between items

**Badge (right):**
- Text: "ShotSync"
- Pink text on white/blur background
- Rounded: 24px
- Links to `./tool/index.html`

---

## 📐 Section 2: HERO

```
┌────────────────────────────────────────────────────────────┐
│                                                            │
│                   [Cause Every Athlete Has Roots!]  ← badge│
│                                                            │
│               Roots - An all in one AI                     │
│               athlete assistant!                           │
│                                                            │
│    We connect athletes, coaches, and content creators      │
│    through intelligent shot creation and comprehensive     │
│    analytics powered by AI.                                │
│                                                            │
│                     [ Try ShotSync ]  ← glassy button      │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**Background layers (bottom to top):**
1. Base color: Light pink `#FFF8F7`
2. Noise texture: 30% opacity grain
3. Diagonal grid pattern: 30% opacity, 60px squares

**Typography:**
- Badge: 14px Courier Prime, coral text
- Title: 64px Forum serif, "Roots" in coral
- Subtitle: 20px Forum, gray text
- Max width: 900px (title), 700px (subtitle)

**Button:**
- Semi-transparent coral with blur
- Padding: 16px × 48px
- Border-radius: 30px (pill shape)
- Hover: Lifts 2px, more opaque

**Height:** 100vh (full viewport)
**Alignment:** Everything centered

---

## 📐 Section 3: HOW IT WORKS

```
┌────────────────────────────────────────────────────────────┐
│                     How it works?                          │
│    Explain how to get started with the product in         │
│    3 simple steps                                          │
│                                                            │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐           │
│  │   1     │      │   2     │      │   3     │← numbered  │
│  │ Choose  │      │ Record  │      │  Get    │  circles   │
│  │  Your   │      │  Your   │      │ Instant │           │
│  │ Player  │      │  Shot   │      │Results  │           │
│  │         │      │         │      │         │           │
│  │ Select  │      │ Use your│      │ View    │           │
│  │ the pro │      │ device  │      │ detailed│           │
│  │ whose...│      │ to rec..│      │ analysis│           │
│  └─────────┘      └─────────┘      └─────────┘           │
└────────────────────────────────────────────────────────────┘
```

**Layout:** 3 columns (responsive: stacks on mobile)

**Each Card:**
- Background: Dark `hsl(var(--muted))`
- Padding: 48px × 32px
- Border-radius: 16px
- Border: 1px solid light pink

**Numbered Circle:**
- Size: 56px × 56px
- Background: Coral `#FF6B7A`
- Color: White
- Font: 24px bold
- Centered

**Text:**
- Heading: 24px Forum, white
- Description: 16px Forum, light gray

---

## 📐 Section 4: DEMO VIDEO

```
┌────────────────────────────────────────────────────────────┐
│              See ShotSync in Action                        │
│       Train like a pro without the pro price tag           │
│                                                            │
│  ┌────────────────────────────────────────────────────┐   │
│  │                                                    │   │
│  │              ▶ VIDEO PLAYER                       │   │
│  │                                                    │   │
│  │           (16:9 aspect ratio)                     │   │
│  │                                                    │   │
│  └────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

**Title:**
- "Shot" has gradient: coral → darker coral
- Rest is normal text
- 48px size

**Video Container:**
- Max-width: 1000px
- Aspect ratio: 16:9 (maintains on resize)
- Border-radius: 16px
- White background with shadow
- Video controls enabled

**Files:**
- `ScreenRecording_11-10-2025 19-48-36_1.mov` (primary)
- `.mp4` fallback

---

## 📐 Section 5: FEATURES (3 Cards)

```
┌────────────────────────────────────────────────────────────┐
│                      Features                              │
│   The recruiting game is broken. We're fixing it.          │
│   One shot at a time, one athlete at a time.               │
│                                                            │
│  ┌──────────┐   ┌──────────────┐   ┌──────────┐          │
│  │    ⚠️    │   │      ✅      │   │    🔭    │          │
│  │  THE     │   │   LIVE NOW   │   │   THE    │          │
│  │ PROBLEM  │   │ ─────────────│   │  VISION  │          │
│  │          │   │  ShotSync    │   │          │          │
│  │ College  │   │ AI Shot      │   │ One AI   │          │
│  │recruiting│   │ Analysis     │   │ Platform │          │
│  │is broken │   │              │   │for Every-│          │
│  │          │   │ • Compare to │   │thing     │          │
│  │ • 73% of │   │   pros       │   │          │          │
│  │   athletes│   │ • Real-time │   │ • AI     │          │
│  │   feel    │   │   feedback  │   │   Coach  │          │
│  │   lost    │   │ • Track     │   │ • Auto   │          │
│  │ • $4.2K/  │   │   progress  │   │   Highlts│          │
│  │   year... │   │ • Improve   │   │ • Smart  │          │
│  │           │   │   your form │   │   Recruit│          │
│  │           │   │             │   │ • etc.   │          │
│  │           │   │[Try Free Now]   │          │          │
│  └──────────┘   └──────────────┘   └──────────┘          │
└────────────────────────────────────────────────────────────┘
```

**Layout:** 3 columns, equal width

**Card Design:**
- Background: Dark `hsl(var(--muted))`
- Padding: 40px
- Border-radius: 16px
- Border: 1px solid light pink

**Middle Card (HIGHLIGHTED):**
- Border: 2px solid coral (thicker!)
- Slightly more prominent
- Has CTA button at bottom

**Structure Each Card:**
1. Icon (64px × 64px from Icons8)
2. Title (18px Courier Prime, uppercase, white)
3. Horizontal line (pink, 80% width)
4. Content section (left-aligned)
   - Bold subtitle
   - Bullet list with coral bullets

**Button (middle card only):**
- Background: Coral `#FF6B7A`
- Text: White
- Full width
- Padding: 16px
- Border-radius: 10px
- Text: "Try Free Now →"

---

## 📐 Section 6: TESTIMONIALS (Twitter Wall)

```
┌────────────────────────────────────────────────────────────┐
│            See What Athletes Are Sharing                   │
│    Athletes worldwide are sharing their similarity scores  │
│    with #rooted                                            │
│                                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │ 👤       │  │ 👤       │  │ 👤       │                │
│  │ @user1   │  │ @user2   │  │ @user3   │                │
│  │          │  │          │  │          │                │
│  │ Just got │  │ 87%      │  │ Training │                │
│  │ my score!│  │ similar! │  │ paying   │                │
│  │ #rooted  │  │ #rooted  │  │ off      │                │
│  │          │  │          │  │ #rooted  │                │
│  │ 2h ago   │  │ 5h ago   │  │ 1d ago   │                │
│  └──────────┘  └──────────┘  └──────────┘                │
│                                                            │
│         Got your similarity score?                         │
│         Share it with the community!                       │
│                                                            │
│           [ Share Your Score on Twitter ]                  │
│                  (Twitter blue button)                     │
└────────────────────────────────────────────────────────────┘
```

**Grid:** Auto-fit columns, min 320px

**Tweet Card:**
- Background: White
- Padding: 24px
- Border-radius: 16px
- Border: 1px solid very light gray
- Hover: Lifts 4px with shadow

**Card Structure:**
1. **Header:**
   - Avatar (48px circle, coral background)
   - Name (15px bold)
   - Handle (14px gray)

2. **Content:**
   - Tweet text (16px)
   - `#rooted` in coral color

3. **Footer:**
   - Date (13px gray)
   - "View on Twitter" link (14px coral)

**Twitter Button:**
- Background: Twitter blue `#1DA1F2`
- Color: White
- Padding: 16px × 48px
- Border-radius: 30px
- Hover: Darker blue, lifts 2px

**Placeholder (when no tweets):**
- Bird emoji
- "Loading tweets with #rooted..."
- Instructions to share

---

## 📐 Section 7: FEEDBACK FORM

```
┌────────────────────────────────────────────────────────────┐
│            We'd Love Your Feedback                         │
│    Help us build the perfect tool for athletes like you    │
│                                                            │
│  ┌────────────────────────────────────────────────────┐   │
│  │                                                    │   │
│  │           TYPEFORM EMBEDDED HERE                  │   │
│  │                                                    │   │
│  │              (600px height)                       │   │
│  │                                                    │   │
│  └────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

**Container:**
- Max-width: 900px
- Background: White
- Padding: 24px
- Border-radius: 16px
- Box-shadow for depth

**Iframe:**
- URL: `https://form.typeform.com/to/wUkR5vhP`
- Width: 100%
- Height: 600px
- No border

---

## 📐 Section 8: FINAL CTA

```
┌────────────────────────────────────────────────────────────┐
│                                                            │
│    ╔════════════════════════════════════════════╗         │
│    ║                                            ║         │
│    ║    Can you shoot like a pro?              ║         │
│    ║                                            ║         │
│    ║    Join thousands of athletes discovering ║         │
│    ║    their roots and improving their game   ║         │
│    ║                                            ║         │
│    ║         [ Get Started Free ]               ║         │
│    ║                                            ║         │
│    ╚════════════════════════════════════════════╝         │
│               (gradient coral box)                         │
└────────────────────────────────────────────────────────────┘
```

**Box:**
- Background: Gradient coral (135deg, light → medium → light)
- Padding: 80px × 60px (60px × 30px on mobile)
- Border-radius: 24px
- Border: 1px solid light coral
- Box-shadow: Large, coral-tinted

**Text:**
- All white color
- Heading: 48px (32px mobile)
- Subheading: 20px
- Centered

**Button:**
- Glassy white style
- Semi-transparent with blur
- White text
- Padding: 18px × 48px
- Border-radius: 12px
- Hover: More opaque, lifts

---

## 📐 Section 9: FOOTER

```
┌────────────────────────────────────────────────────────────┐
│ 🏀 Roots Logo               Privacy                        │
│ © 2025 Roots AI.            Terms        Twitter           │
│ All rights reserved.        Contact      LinkedIn          │
│                                          Instagram          │
└────────────────────────────────────────────────────────────┘
     ↑                            ↑              ↑
   Brand                       Links          Social
```

**Background:** Dark `#1a1a1a`
**Text:** White
**Padding:** 48px vertical

**Layout:** 3 sections horizontal (stacks on mobile)

**Left - Brand:**
- Logo (32px height)
- Copyright text (14px, semi-transparent)

**Center - Links:**
- Privacy, Terms, Contact
- 14px size
- 24px gap
- Semi-transparent, white on hover

**Right - Social:**
- Twitter, LinkedIn, Instagram
- Same styling as links
- Twitter link active: `https://x.com/with__roots`

**Mobile:**
- Stacks vertically
- Centered alignment

---

## 🎨 Hover States (All Interactive Elements)

### Buttons
**Before:** Normal state
**Hover:**
- `transform: translateY(-2px)` - Lifts up
- Increased shadow
- Slightly different background (more opaque or darker)
- 0.3s transition

### Cards
**Before:** Normal state
**Hover:**
- `transform: translateY(-4px)` - Lifts more than buttons
- `box-shadow: 0 12px 32px rgba(0, 0, 0, 0.08)`
- 0.3s transition

### Links
**Before:** Semi-transparent or coral
**Hover:** Full opacity or brighter coral
- 0.3s transition

---

## 📱 Mobile Responsive Behavior

**At 768px width and below:**

```
Desktop (3 columns):          Mobile (1 column):
┌───┬───┬───┐                ┌───────────┐
│ 1 │ 2 │ 3 │                │     1     │
└───┴───┴───┘                ├───────────┤
                              │     2     │
                              ├───────────┤
                              │     3     │
                              └───────────┘
```

**Changes:**
- All grids → single column
- Nav links → hidden (hamburger needed)
- Text sizes reduced ~30%
- Padding reduced
- Footer stacks and centers

---

## 💡 Quick Visual Checklist

When reviewing/implementing, check for:

**Colors:**
- [ ] Coral pink `#FF6B7A` used consistently
- [ ] White → pink gradient on all section backgrounds
- [ ] Dark cards `hsl(222 47% 11%)`

**Typography:**
- [ ] Forum for content
- [ ] Courier Prime for UI
- [ ] Consistent sizes (64/48/24/18/16/14px scale)

**Effects:**
- [ ] Glassmorphism on buttons (blur + transparency)
- [ ] Lift on hover (2-4px up)
- [ ] Smooth 0.3s transitions
- [ ] Noise texture on hero
- [ ] Grid pattern on hero

**Spacing:**
- [ ] 100px section padding (vertical)
- [ ] 1200px max container width
- [ ] 16px card border-radius
- [ ] 30px button border-radius (pills)

**Structure:**
- [ ] 9 sections total
- [ ] Navbar sticky at top
- [ ] All sections have gradient background
- [ ] Footer dark background

---

## 🔗 Important Links to Implement

**CTAs (all go to ShotSync tool):**
```
./tool/index.html
```

**Social:**
```
Twitter: https://x.com/with__roots
```

**Embeds:**
```
Typeform: https://form.typeform.com/to/wUkR5vhP
Icons: https://img.icons8.com/fluency/96/[icon-name].png
```

**Share Intent:**
```
https://twitter.com/intent/tweet?
  text=Just%20got%20my%20shot%20similarity%20score%20from%20ShotSync!%20
  %23rooted%20@with__roots
  &url=https://withroots.org
```

---

## Summary Card

```
╔══════════════════════════════════════════════════════╗
║  ROOTS LANDING PAGE - AT A GLANCE                   ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  Color:     Coral Pink (#FF6B7A)                    ║
║  Fonts:     Forum (serif) + Courier Prime (mono)    ║
║  Effect:    Glassmorphism (blur + transparency)     ║
║  Layout:    9 sections, all with gradient bg        ║
║  Style:     Modern, minimal, warm                   ║
║  Goal:      Convert to ShotSync trial users         ║
║  Audience:  Athletes, coaches, parents              ║
║                                                      ║
║  Key Sections:                                       ║
║  1. Sticky nav                                       ║
║  2. Hero (full screen, grid pattern)                ║
║  3. How it works (3 steps)                          ║
║  4. Demo video (16:9)                               ║
║  5. Features (problem → solution → vision)          ║
║  6. Testimonials (Twitter #rooted)                  ║
║  7. Feedback (Typeform)                             ║
║  8. Final CTA (gradient box)                        ║
║  9. Footer (dark)                                    ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
```

---

**This guide shows the exact visual layout. For detailed specs, see `DESIGN_DOCUMENTATION.md`. For quick reference, see `QUICK_REFERENCE.md`.**
