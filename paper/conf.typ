#let conf(
  title: [],
  authors: (),
  abstract: [],
  doc,
) = {
  place(
    top + center,
    float: true,
    scope: "parent",
    clearance: 2em,
    {
      heading(title, level: 1)

      let count = authors.len()
      let ncols = calc.min(count, 3)
      grid(
        columns: (1fr,) * ncols,
        row-gutter: 24pt,
        ..authors.map(author => [
          #author.name \
          #link("mailto:" + author.email)
        ]),
      )

      par(justify: false)[
        *Abstract* \
        #abstract
      ]

    }
  )

  columns(2)[
    #doc
  ]
}
