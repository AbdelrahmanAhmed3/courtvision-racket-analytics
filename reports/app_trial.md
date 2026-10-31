# Trying the App on New Clips

## The short clips we have (2026-10-31)

#59. The **Analyze a clip** page was run on every short padel clip we have, with its default settings: RF-DETR Nano, the four-ID tracker, TrackNet and the event classifier. The Qatar clip went through the page in the browser. The others were run with the same pipeline command the page uses, with the court marked by the page's six points.

| Clip | Camera | Length | Court marked | Run time | What came out |
| --- | --- | --- | --- | --- | --- |
| Qatar Major, from Daily Padel | broadcast, high, 720p | 15 s | six clicks fit within 0.8 px | about 2 min | as in the README: one rally, 12 shots |
| Tapia highlights, from Premier Padel | broadcast, high; a screen recording of the YouTube player, with its buttons over the video; three camera cuts | 31 s | from the labels, scaled to the recording | 3 min 45 s | the first point is right; everything after the first cut is wrong |
| "The best point in padel history", World Padel Tour | broadcast, high, 480p | 77 s | by eye, six points within 1.4 px | 9 min 21 s | one 74.8 s rally, 44 shots, four players kept |
| "What a padel point", club | low, behind the back glass, 720p | 49 s | **could not be marked** | – | – |

Analysis takes about 7 to 8 seconds per second of video on an Apple M4.

### What works

- **Marking the court.** Six clicks with the guide take under a minute, and the fit check catches a misclick. During the test the author found that "Undo" brought the last click back; that is fixed in #59.
- **Player identity over a long point.** On the 77-second WPT point, the four IDs stay on the same players throughout, including several side switches: checked at 10, 40 and 70 s. On the court map, partners stay on their own half.
- **Hits on a messier video.** On the Tapia clip's first point the hits were scored against the labels, after aligning the screen recording to the labelled video frame by frame. The recording's clock drifts by up to 2 s in its first 10 s. Hits were found 6 of 16 times, and 6 of the 19 predicted were real. That's close to the measured recall 0.38 and precision 0.44, with the player's buttons drawn over the video. These rallies are in the labelled set, so this checks robustness, not unseen data.

### What breaks

1. **A clip with camera cuts, as the known limitations warn.** The Tapia clip cuts to a replay from a side camera, then to a point on another court in Brussels. The one calibration is applied to every frame:
   - the replay's players land on the court map as if seen from the main camera;
   - the Brussels point is mapped with the wrong court;
   - the report joins everything into one 29.4 s "rally" with 20 shots.

   CourtVision analyzes one continuous view of one court: the app asks for a clip "filmed from one high camera, without cuts", and the README lists it as a known limitation. Compilations are out of scope; their first point, before any cut, comes out right.
2. **Impossible shot speeds, fixed in #59.** The "fastest shot" was 232 km/h on the WPT clip and 183.5 km/h on the Tapia clip. Both came from pairs of hits that can't be one shot:
   - a floor distance longer than the court's diagonal (24.2 and 21.1 m on a 20 m court), meaning the ball was in the air and projected far off the court;
   - a shot "received" by the hitter's own partner;
   - a receiving hit with no player near it.

   One rule, `has_credible_speed`, now keeps these out of both the court map's labels and the report's fastest shot. The WPT fastest shot is now 135.4 km/h.
3. **A low camera behind the glass ([#32](https://github.com/AbdelrahmanAhmed3/courtvision-racket-analytics/issues/32)).** From player height behind the back glass, the near service line runs off both edges of the frame. Its ends, two of the six points, can't be clicked. This is the Personal edition's camera, so it needs another way to mark the court: other landmarks, or the line directions instead of their ends.
4. **Hit detection on 480p.** On the WPT clip, 18 of the 43 consecutive pairs of shots have the same team hitting twice. On the Qatar clip it's 4 of 11, and on Tapia 5 of 19. The ball is a few pixels wide at 480p, and the TrackNet ball tracker, trained on tennis, loses it more often. This adds weight to [#61](https://github.com/AbdelrahmanAhmed3/courtvision-racket-analytics/issues/61) (a padel ball tracker) and the swing trial in [#60](https://github.com/AbdelrahmanAhmed3/courtvision-racket-analytics/issues/60). The same-team count gives a measure on clips without labels.
