# Frontend Redesign Plan

## Objective
Convert the basic Voice RAG tester into a professional landing page with:
1. Split-screen hero (left: headline/CTA, right: flower image + glassmorphism)
2. ProductHunt badge #1
3. AI assistant chat UI demo
4. Voice transcription feature highlight
5. Logo bar (Adobe, Intel, Google, etc.)

## Current State
- `app/static/index.html`: 289 lines, basic functional tester
- `index.html`: 316 lines, duplicate at root
- No branding, no marketing sections, no visual design system

## Proposed Structure

### Section 1: Split-Screen Hero
- Left: Headline, subheadline, CTA buttons (Try Demo, View Docs), trust badges
- Right: Abstract generative flower visual using CSS/SVG with glassmorphism overlay
- ProductHunt badge positioned as a floating element

### Section 2: AI Chat Demo
- Mock chat interface showing voice RAG in action
- Animated typing effect
- Voice waveform visualization
- "Try it yourself" embedded live demo below

### Section 3: Features Grid
- Voice Transcription (Sarvam + Whisper)
- RAG Pipeline
- Multilingual Support (Hindi, Bengali, English)
- Guardrails & Safety

### Section 4: Logo Bar
- Adobe, Intel, Google, Microsoft, NVIDIA (text/SVG logos)
- "Trusted by teams at" label

### Section 5: Footer
- Links, version, status

## Tech Stack
- Tailwind CSS via CDN (no build step)
- Vanilla JS (no framework overhead)
- CSS-only animations
- SVG for logos and decorative elements

## File Plan
1. `app/static/index.html` — new landing page
2. `app/static/demo.js` — chat demo interactions
3. Keep `/transcribe`, `/query`, `/voice-query` functional for live demo

## Design Tokens
- Primary: Sky blue (#0ea5e9)
- Accent: Violet (#8b5cf6)
- Background: Slate 900 gradient
- Glass: rgba(255,255,255,0.05) with backdrop-blur
- Font: Inter (Google Fonts)
