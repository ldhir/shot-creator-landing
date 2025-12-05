# Roots Landing Page - Design Documentation

**Product:** ShotSync by Roots AI
**Tagline:** "Cause Every Athlete Has Roots!"
**Purpose:** AI-powered shot analysis tool for athletes

---

## Table of Contents
1. [Overview](#overview)
2. [Design System](#design-system)
3. [Page Structure](#page-structure)
4. [Section-by-Section Breakdown](#section-by-section-breakdown)
5. [Interactive Elements](#interactive-elements)
6. [Responsive Design](#responsive-design)
7. [Technical Implementation](#technical-implementation)

---

## Overview

### Product Description
**Roots** is an all-in-one AI athlete assistant platform. The landing page focuses on their first product: **ShotSync** - an AI shot analysis tool that allows athletes to compare their shooting form to professional players.

### Target Audience
- Athletes (primarily basketball players)
- Coaches looking for training tools
- Parents of young athletes
- Content creators in sports

### Key Value Propositions
1. **Solve the Recruiting Problem** - College recruiting is broken, expensive, and fragmented
2. **ShotSync (Live Now)** - AI shot analysis comparing your form to pros
3. **Future Vision** - One AI platform for everything (recruiting, highlights, coaching)

---

## Design System

### Color Palette

**Primary Colors:**
```css
--primary-color: #FF6B7A (Coral Pink - main brand color)
--primary-hover: #FF5468 (Darker coral for hover states)
```

**Background Gradient:**
```css
background: linear-gradient(90deg,
    rgba(255, 255, 255, 1) 0%,
    rgba(255, 179, 186, 1) 85%,
    rgba(255, 107, 122, 1) 100%
);
```
White fading into soft pink, then into coral pink - used throughout all sections.

**Text Colors:**
```css
--text-dark: #1a1a1a (Primary text)
--text-gray: #666666 (Secondary text)
```

**Card/Surface Colors:**
```css
--bg-white: #FFFFFF
--bg-light: #FAFAFA
Dark cards: hsl(222 47% 11%) - Dark blue-gray
```

### Typography

**Primary Font:** `Forum` (Serif) - Used for headings and body text
```css
font-family: 'Forum', -apple-system, BlinkMacSystemFont, 'Segoe UI', serif;
```

**Secondary Font:** `Courier Prime` (Monospace) - Used for buttons, badges, nav links
```css
font-family: 'Courier Prime', monospace;
```

**Font Sizes:**
- Hero Title: 64px (large screens) → 40px (mobile)
- Section Titles: 48px → 32px (mobile)
- Hero Subtitle: 20px → 16px (mobile)
- Body Text: 15-18px
- Small Text: 13-14px

### Spacing System
- Section Padding: 100px vertical
- Container Max-Width: 1200px
- Standard Gap: 24-40px
- Card Padding: 40-48px

### Visual Effects

**Glassmorphism (Glassy Buttons):**
```css
background: rgba(255, 255, 255, 0.1);
backdrop-filter: blur(12px);
border: 1px solid rgba(255, 255, 255, 0.2);
box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
```

**Noise Texture (Hero Section):**
- SVG fractal noise overlay at 30% opacity
- Adds subtle grain texture to background

**Grid Pattern (Hero):**
```css
background-image:
    linear-gradient(65deg, rgba(255, 107, 122, 0.15) 1px, transparent 1px),
    linear-gradient(-25deg, rgba(255, 107, 122, 0.15) 1px, transparent 1px);
background-size: 60px 60px;
```
Diagonal grid pattern in brand color at 30% opacity.

---

## Page Structure

### Navigation Sections (Anchors)
1. `#demo` - Demo video section
2. `#benefits` - Features/Benefits section
3. `#faq` - Feedback form section

### Complete Page Flow
```
├── Navbar (Sticky)
├── Hero Section
├── How It Works (3 Steps)
├── Demo Video
├── Benefits/Features (Problem → Solution → Vision)
├── Testimonials (Twitter Wall)
├── Feedback Form (Typeform)
├── CTA (Final Call to Action)
└── Footer
```

---

## Section-by-Section Breakdown

### 1. Navbar

**Layout:** Horizontal flexbox
**Position:** Sticky (stays at top while scrolling)
**Background:** Semi-transparent white with blur effect

**Elements:**
- **Left:** Roots logo (50px height)
- **Center:** Navigation links
  - "Get Demo" → scrolls to #demo
  - "About" → scrolls to #benefits
  - "Questions" → scrolls to #faq
- **Right:** "ShotSync" badge (links to ./tool/index.html)

**Styling:**
- Glassmorphic background: `rgba(255, 255, 255, 0.8)` with `backdrop-filter: blur(10px)`
- Border bottom: Subtle pink border `rgba(255, 107, 122, 0.1)`
- Box shadow for depth

**Mobile Behavior:**
- Nav links hidden on screens < 768px
- Logo and badge remain visible

---

### 2. Hero Section

**Purpose:** First impression, main value proposition

**Layout:**
- Full viewport height (`min-height: 100vh`)
- Centered content vertically and horizontally
- Layered background effects

**Content:**
```
├── Badge: "Cause Every Athlete Has Roots!"
├── Headline: "Roots - An all in one AI athlete assistant!"
├── Subtitle: Connection statement about athletes, coaches, creators
└── CTA Button: "Try ShotSync" → links to tool
```

**Background Layers (bottom to top):**
1. Light pink base color `#FFF8F7`
2. SVG noise texture (30% opacity) - adds grain
3. Diagonal grid pattern (30% opacity) - brand identity
4. Canvas element (hidden/display:none) - placeholder for future animation

**Typography:**
- "Roots" in headline is highlighted in brand color (#FF6B7A)
- Large, bold serif font creates impact
- Subtitle in gray for visual hierarchy

**CTA Button:**
- Glassy style with pink tint
- Large padding: 16px × 48px
- Rounded: 30px border-radius
- Hover: Lifts up 2px with shadow

---

### 3. How It Works

**Purpose:** Explain the product in 3 simple steps

**Layout:** 3-column grid (responsive)
```
[Step 1] [Step 2] [Step 3]
```

**Each Step Card Contains:**
1. **Step Number** - Circular badge (56px) with number
   - Background: Primary coral color
   - White text, centered
2. **Heading** - Action-oriented title
3. **Description** - Brief explanation

**Steps:**
1. **Choose Your Player** - Select pro to emulate
2. **Record Your Shot** - Film your form from same angle
3. **Get Instant Results** - View analysis and comparison

**Card Styling:**
- Dark cards: `hsl(var(--muted))` - dark blue-gray
- Rounded corners: 16px
- Padding: 48px × 32px
- Centered text alignment

**Grid Behavior:**
- Desktop: 3 columns
- Mobile: Stacks to 1 column

---

### 4. Demo Video Section

**Purpose:** Show the product in action

**Content:**
- Section title: "See **Shot**Sync in Action"
  - "Shot" has gradient text effect in brand colors
- Subtitle: "Train like a pro without the pro price tag"
- Video player

**Video Container:**
- Max width: 1000px
- 16:9 aspect ratio maintained with padding-bottom hack
- Rounded corners: 12px
- Box shadow for depth
- White background

**Video Sources:**
- `.mov` format (QuickTime)
- `.mp4` fallback
- Controls enabled
- Preload metadata only (performance)

**Styling Notes:**
- Video fills container with `object-fit: contain`
- Fallback message for unsupported browsers

---

### 5. Benefits/Features Section

**Purpose:** Show problem → solution → vision

**Layout:** 3-column grid with center card highlighted

```
[Problem Card] [Solution Card - Highlighted] [Vision Card]
```

**Card Types:**

#### A. THE PROBLEM Card
- **Icon:** Error/problem icon (red)
- **Content:** "College recruiting is broken"
- **Bullet Points:**
  - 73% of athletes feel lost
  - $4.2K/year across 5-10 platforms
  - 500K athletes → 180K NCAA spots
  - No centralized platform
  - Expensive recruiting services
  - Coaches can't find talent
  - Talented athletes miss opportunities

#### B. LIVE NOW Card (Highlighted)
- **Icon:** Checkmark icon (green)
- **Title:** "ShotSync - AI Shot Analysis"
- **Features:**
  - Compare to pros
  - Real-time feedback
  - Track progress
  - Improve your form
- **CTA Button:** "Try Free Now →"
- **Special Styling:**
  - 2px coral border
  - Slightly elevated with shadow
  - Draws most attention

#### C. THE VISION Card
- **Icon:** Telescope icon (blue)
- **Content:** "One AI Platform for Everything"
- **Future Features:**
  - AI Coach (JARVIS)
  - Auto Highlights
  - Smart Recruiting
  - Vetted Marketplace
  - Voice AI Coach
  - NIL Management
  - NCAA Messaging

**Card Styling:**
- Dark background: `hsl(var(--muted))`
- White/light text
- Icon at top (64px)
- Horizontal divider below title
- Left-aligned bullet points
- Coral bullet points
- Hover effect: Lifts 4px with shadow

---

### 6. Testimonials - Twitter Wall

**Purpose:** Social proof through user-generated content

**Layout:** Grid of tweets (auto-fit, min 320px columns)

**Content:**
- Section title: "See What Athletes Are Sharing"
- Subtitle: "Athletes worldwide are sharing their similarity scores with **#rooted**"
- Tweet grid container
- Call-to-action button

**Tweet Card Structure:**
```
├── Header
│   ├── Avatar (48px circle)
│   ├── Author name (bold)
│   └── @handle (gray)
├── Tweet content (with #rooted hashtag highlighted)
├── Footer
│   ├── Date
│   └── "View on Twitter" link
```

**Twitter CTA:**
- Text: "Got your similarity score? Share it with the community!"
- Button: "Share Your Score on Twitter"
  - Twitter blue background (#1DA1F2)
  - Pre-populated tweet with #rooted @with__roots
  - Opens Twitter intent in new tab

**Placeholder State:**
- Shows when no tweets loaded
- Bird emoji + loading message
- Instructions to share with #rooted

**Tweet Styling:**
- White cards
- Rounded: 16px
- Subtle border
- Hover: Lift and shadow
- Coral color for hashtags

**Integration:**
- JavaScript file: `twitter-wall.js`
- Fetches tweets with #rooted hashtag
- Real-time social proof

---

### 7. Feedback Section

**Purpose:** Collect user feedback

**Content:**
- Section title: "We'd Love Your Feedback"
- Subtitle: "Help us build the perfect tool for athletes like you"
- Embedded Typeform

**Typeform Integration:**
- URL: `https://form.typeform.com/to/wUkR5vhP`
- Full-width iframe
- Height: 600px
- Rounded container with white background
- Box shadow for depth

**Container Styling:**
- Max width: 900px
- Centered
- White background
- Padding: 24px
- Rounded: 16px

---

### 8. CTA Section (Final)

**Purpose:** Last chance conversion

**Content:**
- Headline: "Can you shoot like a pro?"
- Subheading: "Join thousands of athletes discovering their roots and improving their game"
- CTA Button: "Get Started Free"

**Styling:**
- **Background:** Gradient box with coral color
```css
background: linear-gradient(135deg,
    rgba(255, 107, 122, 0.8) 0%,
    rgba(255, 138, 149, 0.9) 50%,
    rgba(255, 107, 122, 0.8) 100%
);
```
- White text
- Large padding: 80px × 60px
- Rounded: 24px
- Border and shadow for depth

**Button:**
- Glassy white style
- Semi-transparent
- Blur effect
- Hover: More opaque, lifts up

---

### 9. Footer

**Layout:** 3-section horizontal layout

**Sections:**
```
[Brand/Logo]  [Links]  [Social Media]
```

**Content:**

**Left - Brand:**
- Roots logo (32px)
- Copyright: "© 2025 Roots AI. All rights reserved."

**Center - Links:**
- Privacy
- Terms
- Contact

**Right - Social:**
- Twitter (https://x.com/with__roots)
- LinkedIn
- Instagram

**Styling:**
- Dark background: `var(--text-dark)` (#1a1a1a)
- White text
- Semi-transparent link colors
- Hover: Full white
- Padding: 48px vertical

**Mobile Behavior:**
- Stacks vertically
- Centers all content

---

## Interactive Elements

### Buttons

**1. Glassy Primary (Hero CTA)**
```css
background: rgba(255, 107, 122, 0.1);
backdrop-filter: blur(12px);
border: 1px solid rgba(255, 107, 122, 0.3);
color: coral pink;
padding: 16px 48px;
border-radius: 30px;
```
**Hover:** More opaque background, lifts 2px

**2. Glassy White (Final CTA)**
```css
background: rgba(255, 255, 255, 0.1);
backdrop-filter: blur(12px);
border: 1px solid rgba(255, 255, 255, 0.2);
color: white;
```
**Hover:** More opaque, lifts 2px

**3. Plan Button (Features Section)**
```css
background: coral pink;
color: white;
border-radius: 10px;
padding: 16px;
```
**Hover:** Lifts 2px with shadow

**4. Twitter Button**
```css
background: #1DA1F2 (Twitter blue);
color: white;
border-radius: 30px;
```
**Hover:** Darker blue, lifts 2px

### Cards

**All Cards Share:**
- Border-radius: 16px
- Transition effects on transform and shadow
- Hover state: `translateY(-4px)` with increased shadow

**Types:**
1. **Step Cards** - Dark background, white text, numbered badges
2. **Benefit Cards** - Dark background, icons, bullet lists
3. **Tweet Cards** - White background, profile info, content

### Navigation

**Smooth Scroll:**
- Links use `href="#section-id"`
- Smooth scrolling behavior (browser default or JavaScript)

**Sticky Navbar:**
- `position: sticky; top: 0;`
- Remains visible while scrolling
- Subtle shadow for depth

---

## Responsive Design

### Breakpoint: 768px

**Changes at < 768px:**

**Navigation:**
- Hide nav links (`.nav-links { display: none; }`)
- Keep logo and ShotSync badge

**Typography:**
- Hero title: 64px → 40px
- Hero subtitle: 20px → 16px
- Section titles: 48px → 32px
- Section subtitles: 18px → 16px

**Layout:**
- All grids: Multi-column → Single column
  - Benefits grid
  - Steps grid
  - Pricing grid
  - Testimonials grid

**CTA Section:**
- Padding: 80px 60px → 60px 30px
- Heading: 48px → 32px

**Footer:**
- Flex direction: Row → Column
- Text align: Left → Center
- Brand section: Centered

**Video:**
- Maintains 16:9 aspect ratio
- Responsive width

---

## Technical Implementation

### File Structure
```
shot-creator-landing/
├── index.html          # Main landing page
├── styles.css          # All styles
├── script.js           # Main JavaScript
├── twitter-wall.js     # Twitter integration
├── RootsLogo.png       # Brand logo
├── .env                # Environment variables
└── tool/               # ShotSync tool directory
    └── index.html      # Actual tool
```

### Key Technologies

**HTML5:**
- Semantic sections
- Video element with multiple sources
- Iframe for Typeform
- Canvas elements (placeholders)

**CSS3:**
- CSS Grid and Flexbox
- CSS Variables (custom properties)
- Backdrop-filter (glassmorphism)
- Linear gradients
- SVG data URLs (noise texture)
- Media queries

**JavaScript:**
- `script.js` - General interactivity
- `twitter-wall.js` - Twitter API integration
- Canvas animations (optional)

**External Integrations:**
- Google Fonts (Forum, Courier Prime)
- Typeform (feedback form)
- Twitter (social sharing, tweet wall)
- Icons8 (icon images)

### Performance Optimizations

**Images:**
- Logo optimized PNG
- Icon images from CDN (Icons8)
- Video preload: metadata only

**Fonts:**
- Google Fonts with `preconnect`
- Font-display: swap

**Layout:**
- GPU-accelerated transforms (translateY)
- Will-change: transform (for animations)
- Debounced scroll events

**Loading:**
- Video lazy-loaded
- Iframe lazy-loaded (Typeform)
- CSS loaded in head
- JS loaded at end of body

---

## Design Principles

### 1. Visual Hierarchy
- Large hero section grabs attention
- Clear section titles
- Progressive disclosure (problem → solution → vision)

### 2. Consistency
- Coral pink (#FF6B7A) used throughout for brand recognition
- Consistent spacing (100px section padding)
- Uniform border-radius (16px for cards, 30px for buttons)
- Same gradient background on all sections

### 3. Modern Aesthetics
- Glassmorphism creates depth
- Noise texture adds sophistication
- Grid pattern reinforces brand
- Smooth transitions and hover effects

### 4. Trust Building
- Social proof (Twitter wall)
- Clear problem statement
- Professional design
- Real product demo

### 5. Conversion Focused
- Multiple CTAs throughout
- Free trial emphasis
- Low-friction entry (just try it)
- Clear value proposition

---

## Key User Flows

### 1. Quick Trial Flow
```
Navbar "ShotSync" → Tool
Hero CTA → Tool
Benefits "Try Free Now" → Tool
Final CTA → Tool
```

### 2. Learning Flow
```
Hero → How It Works → Demo Video → Benefits → Try Tool
```

### 3. Social Proof Flow
```
Demo → Testimonials (Twitter) → Share CTA → User tweets
```

### 4. Feedback Flow
```
Any section → "Questions" in nav → Typeform → Submit feedback
```

---

## Content Strategy

### Messaging Hierarchy

**Primary Message:**
"Roots - An all in one AI athlete assistant"

**Secondary Messages:**
- Cause Every Athlete Has Roots! (tagline)
- Train like a pro without the pro price tag (demo section)
- The recruiting game is broken. We're fixing it. (benefits)

### Call-to-Action Progression
1. **Hero:** "Try ShotSync" - Low commitment, exploratory
2. **Benefits:** "Try Free Now →" - Emphasizes free, directional arrow
3. **Final CTA:** "Get Started Free" - Strong action verb, commitment

### Social Proof Strategy
- Twitter wall with #rooted
- Share button pre-populated with message
- User-generated content visibility

---

## Brand Guidelines

### Voice & Tone
- **Empowering:** "Can you shoot like a pro?"
- **Friendly:** "We'd love your feedback"
- **Direct:** "The recruiting game is broken. We're fixing it."
- **Inclusive:** "Every Athlete Has Roots"

### Visual Identity
- **Primary Color:** Coral Pink (#FF6B7A) - Energetic, warm, approachable
- **Typography:** Mix of elegant serif (Forum) with technical monospace (Courier Prime)
- **Effects:** Glass morphism suggests modern, cutting-edge technology
- **Patterns:** Grid suggests data, analysis, precision

### Logo Usage
- Height: 50px (navbar), 32px (footer)
- Always on white/light backgrounds
- Maintains aspect ratio
- Clear space around logo

---

## Accessibility Considerations

**Current Implementation:**
- Semantic HTML5 elements
- Alt text on logo images
- Sufficient color contrast (needs testing)
- Focus states on interactive elements
- Keyboard navigation for nav links

**Improvements Needed:**
- Alt text for all images
- ARIA labels for navigation
- Skip to content link
- Keyboard navigation for video player
- Focus indicators for all interactive elements
- Color contrast validation (WCAG AA)

---

## Future Enhancements

### Phase 1 (Immediate)
- [ ] Mobile hamburger menu
- [ ] Loading states for video/Typeform
- [ ] Analytics integration
- [ ] A/B testing setup

### Phase 2 (Near-term)
- [ ] Animated hero canvas background
- [ ] Scroll animations (reveal on scroll)
- [ ] Live Twitter integration
- [ ] Newsletter signup

### Phase 3 (Long-term)
- [ ] Multi-language support
- [ ] Dark mode toggle
- [ ] Video testimonials
- [ ] Interactive demo without leaving page

---

## Analytics & Tracking Points

**Key Events to Track:**
1. Hero CTA clicks
2. Benefits CTA clicks
3. Final CTA clicks
4. Video play/completion
5. Twitter share clicks
6. Typeform submissions
7. Nav link clicks
8. External link clicks (social, etc.)
9. Time on page
10. Scroll depth

---

## SEO Considerations

**Meta Information:**
- Title: "ShotSync by Roots AI - Cause Every Athlete Has Roots!"
- Description: (needs to be added)
- Keywords: athlete training, shot analysis, AI coaching, basketball training

**Structured Data:**
- Product schema (for ShotSync)
- Organization schema (for Roots AI)
- Video object (for demo)

**Content SEO:**
- H1: Main headline
- H2: Section titles
- H3: Card titles
- Semantic HTML structure
- Internal linking (to tool)

---

## Browser Compatibility

**Tested On:**
- Chrome (primary)
- Safari (primary - Mac users)
- Firefox
- Edge

**CSS Features That Need Fallbacks:**
- `backdrop-filter` (not supported in old Firefox)
- CSS Grid (needs flexbox fallback for IE11)
- CSS Variables (needs static fallbacks for IE11)

**Graceful Degradation:**
- Video: MP4 fallback + message for unsupported browsers
- Fonts: System font fallback stack
- Effects: Basic styles without blur/glass effects

---

## Performance Metrics

**Target Metrics:**
- First Contentful Paint (FCP): < 1.8s
- Largest Contentful Paint (LCP): < 2.5s
- Time to Interactive (TTI): < 3.8s
- Cumulative Layout Shift (CLS): < 0.1

**Current Optimizations:**
- Minified CSS (needs to be done)
- Optimized images
- Lazy-loaded video
- Font preconnect

---

## Maintenance Notes

**Regular Updates:**
- Update Twitter wall content
- Refresh demo video seasonally
- Update testimonials
- Keep feature list current (as product evolves)

**Monitoring:**
- Check video playback across devices
- Test Typeform integration
- Verify all CTAs link correctly
- Monitor page load speed

**Content Updates:**
- Update stats in "Problem" section as data changes
- Add new features to "Vision" section as they're built
- Update copyright year in footer

---

## Summary

This landing page uses a modern, minimalist design with:
- Coral pink brand color throughout
- Glassmorphic effects for modern feel
- Clear problem → solution → vision narrative
- Multiple conversion points
- Social proof integration
- Mobile-responsive layout

The design balances aesthetic appeal with conversion optimization while maintaining brand identity through consistent use of color, typography, and visual effects.
