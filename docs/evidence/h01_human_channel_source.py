"""Select identified human source records without discarding their QC flags."""


def select_human_records(records, *, require_species_column=False):
    """Separate human rows using the published source identity convention.

    Parameters
    ----------
    records : iterable of dict
        Original workbook rows with Filename and optional Species fields.
    require_species_column : bool, optional
        Require an explicit Human/Mouse label agreeing with the filename.

    Returns
    -------
    tuple of list and dict
        Copies of human records and selection counts. Missing values and QC
        flags remain unchanged; this operation does not perform physiological QC.
    """
    human, excluded, seen = [], [], set()
    for record in records:
        name = record.get('Filename')
        if not isinstance(name,str) or not name or name[0] not in ('H','M') or name in seen:
            raise ValueError('Missing, ambiguous or duplicate source identity.')
        seen.add(name)
        expected = {'H':'Human','M':'Mouse'}[name[0]]
        explicit = record.get('Species')
        if (require_species_column and explicit is None) or (explicit is not None and explicit != expected):
            raise ValueError('Explicit species disagrees with the documented filename rule.')
        if expected == 'Human':human.append(dict(record))
        else:excluded.append(name)
    if not human:
        raise ValueError('No identified human source records.')
    return human, dict(human_records=len(human),excluded_nonhuman_records=len(excluded),
        excluded_nonhuman_identities=excluded,
        species_rule='explicit species and H/M filename agree' if require_species_column else 'documented H/M filename prefix',
        physiological_qc_applied=False)
