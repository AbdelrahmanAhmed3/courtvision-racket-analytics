# CourtVision is AGPL-3.0, with commercial licences available

CourtVision is open source so it can be read, learned from and used, but the author does not want a company to take it into a closed product without giving back or paying. We license it under the GNU AGPL-3.0 (only): anyone may use and modify it, but anyone who distributes it or offers it as a network service must publish their whole product's source under the AGPL. Because the author holds all the copyright, the author can also sell a separate commercial licence to anyone who wants to use CourtVision without those obligations (dual licensing).

## Considered Options

- **Apache-2.0**: rejected. Most adoption, but any company could build a closed product on it.
- **Source-available, non-commercial (PolyForm Noncommercial, Business Source License)**: rejected. Not open source, which hurts trust, contributions and the portfolio.
- **No licence**: rejected. Nobody could legally use it, which defeats the portfolio and adoption goals.

## Consequences

- Some companies avoid AGPL code, so adoption by companies will be lower than with Apache-2.0.
- Dual licensing only works while the author owns all the copyright: external contributions need a contributor licence agreement before they are merged.
- Dependencies must stay permissive (ADR 0002). Any AGPL or non-commercial dependency, such as Ultralytics, would make a commercial licence impossible without its own paid licence.
- Model weights CourtVision publishes (v4.1) get their own licence decision when they are released.
