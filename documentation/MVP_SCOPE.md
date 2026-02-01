# MVP Scope - What's In & What's Out

**Version:** 1.0
**Date:** February 2026
**Status:** Final - Ready for Engineering

---

## ✅ IN SCOPE (v1.0)

### Core Gameplay
- [x] 4-player multiplayer only
- [x] All 7 standard combinations
- [x] All 9 bomb types with correct ranking
- [x] Wild cards (hearts of current level)
- [x] Tribute system (2nd hand onwards)
- [x] Level progression (2-A)
- [x] 30-second turn timer with +5s additions (no penalties)
- [x] Bombs playable only on your turn

### Multiplayer
- [x] Room creation with 6-char code
- [x] Room joining by code
- [x] Manual team selection (NS/EW)
- [x] Real-time game state sync (Supabase)
- [x] Disconnect detection (30s grace period)
- [x] Game cancellation on timeout

### UI/UX
- [x] Portrait mode only
- [x] 30-second poker-style timer
- [x] Card sorting (by rank, by suit)
- [x] Straight flush finder (suit buttons with cycling)
- [x] Wild card display (gold border)
- [x] Play/Pass/+Time buttons
- [x] Animal emoji avatars (9 choices: 🦊🐼🦁🐨🐯🐸🐷🐵🐔)

### Technical
- [x] React Native + Expo (SDK 54)
- [x] Supabase (PostgreSQL + Realtime)
- [x] Zustand state management
- [x] Anonymous authentication (display name only)
- [x] EAS build pipeline
- [x] TestFlight deployment
- [x] SF Pro system font (no custom fonts)
- [x] Open-source card SVG assets

---

## ❌ OUT OF SCOPE (v1.1+)

### Features
- [ ] AI opponents / single-player mode
- [ ] In-game chat
- [ ] Emoji reactions
- [ ] Game history / replays
- [ ] Leaderboards / rankings
- [ ] Achievements / badges
- [ ] Friend system
- [ ] Spectator mode
- [ ] Bombs playable at any time (interrupt mechanic)

### Polish
- [ ] Dark mode
- [ ] Landscape orientation
- [ ] Sound effects
- [ ] Haptic feedback
- [ ] Advanced animations (beyond basic dealing)
- [ ] Custom card designs
- [ ] Profile customization
- [ ] Time addition penalties

### Technical
- [ ] Email/password authentication
- [ ] Cross-platform (Android)
- [ ] Offline mode
- [ ] Game state recovery (crashed games)
- [ ] Advanced analytics
- [ ] Push notifications
- [ ] Account system with login

---

## 🎯 MVP Success Criteria

### Gameplay Quality
- [ ] All card combinations validate correctly (100% accuracy)
- [ ] Multiplayer sync <500ms latency
- [ ] Zero game-breaking bugs in 20 test games
- [ ] Tribute system works correctly for all win scenarios
- [ ] Leader selection based on tribute works correctly

### User Experience
- [ ] Room join takes <30 seconds
- [ ] Card selection feels natural (touch feedback)
- [ ] 30-second timer is fair (no false timeouts)
- [ ] Disconnect handling is clear to users
- [ ] Wild cards are easily identifiable (gold border)

### Technical Performance
- [ ] App size <50MB
- [ ] App launch <3 seconds
- [ ] 60 FPS card animations
- [ ] No memory leaks in 1-hour session
- [ ] Supabase free tier sufficient for beta (<100 users)

### Beta Feedback
- [ ] 8-12 testers from friends/family network
- [ ] Complete 5+ full games to Ace level
- [ ] Collect feedback via form
- [ ] Iterate on top 3 pain points

---

## 📋 Decision Summary

All critical decisions have been finalized:

| Decision | Outcome |
|----------|---------|
| Timer duration | 30 seconds |
| Bomb timing | Only on your turn |
| Time penalties | None (MVP) |
| Disconnect handling | Cancel game after 30s |
| Authentication | Anonymous with display name |
| Card grouping storage | Client-side only |
| Straight flush finder | Suit buttons + manual cycle |
| Wild card display | Gold border |
| Font | SF Pro (iOS system) |
| Card assets | Open-source SVG deck |
| Avatars | 9 animal emojis |
| Infrastructure | Supabase free tier |
| Beta testing | Friends/family network |

---

## 🚧 Known Limitations

1. **No AI opponents** - Requires 4 human players
2. **No communication** - Players must use external chat/video
3. **No game recovery** - Disconnects cancel the game
4. **Portrait only** - No landscape support
5. **iOS only** - Android in v1.1+
6. **Anonymous accounts** - No cross-device sync
7. **No dark mode** - Light mode only
8. **English/Chinese only** - No localization
9. **Basic animations** - No advanced effects
10. **No sound** - Silent gameplay

---

## 🔄 Version Roadmap

### v1.0 (MVP) - 4-5 weeks
- All features listed in "IN SCOPE" above
- TestFlight beta with 8-12 testers
- Bug fixes from beta feedback

### v1.1 (Polish) - 2-3 weeks after v1.0
- Sound effects & haptic feedback
- Dark mode
- Improved animations
- Performance optimizations

### v1.2 (Features) - 4-6 weeks after v1.1
- Game history/stats
- Email/password authentication
- Landscape orientation
- Android version

### v2.0 (Social) - Future
- Friend system
- Leaderboards
- Achievements
- In-game chat
- Spectator mode

---

*MVP Scope Documentation v1.0 - February 2026*
