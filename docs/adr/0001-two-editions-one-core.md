# Two editions share one core in one repository

CourtVision serves two very different kinds of footage: high-angle or broadcast clips where the whole court is visible (the Showcase edition), and a fixed low camera behind one back glass where only the near-side players are reliable (the Personal edition). The editions need different analytics, but calibration, detection, tracking, projection and ball events are the same problem in both. We keep one repository with a shared core package and put the edition-specific analytics and apps on top, instead of forking the Personal edition into its own repo.

## Consequences

- A fix to the core reaches both editions; an edition may only add analytics, never patch core behaviour locally.
- The Showcase edition is built first (v4.x). The Personal edition starts once the core has proven itself on showcase clips.
