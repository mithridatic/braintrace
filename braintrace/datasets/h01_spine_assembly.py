"""Preserve source H01 selections while adding explicit electrical spines."""

from dataclasses import dataclass, asdict

from braincell.filter import AtLocation

from braintrace.biophysics.spines import add_spines
from braintrace.biophysics.spine_mapping import remap_location, remap_electrical_regions
from .h01_anatomy import _Region, _geometry_signature


@dataclass(frozen=True)
class H01SpineAssembly:
    """New cable geometry plus source remapping, without rewriting its source.

    Attributes
    ----------
    morphology : braincell.Morphology
        Subdivided source cables and explicit neck/head branches.
    regions : dict
        Complete geometry-bound electrical partition.
    soma : LocsetExpr
        Remapped source soma used for the probe and current input.
    evidence : dict
        Source hash, immutable additions, source map and assembled geometry hash.
    """

    morphology: object
    regions: dict
    soma: object
    evidence: dict


def assemble_spines(imported, regions, spines):
    """Build explicit additions and remap all source electrical selections.

    Parameters
    ----------
    imported : H01Component
        Original measured source, left unchanged.
    regions : dict
        Valid complete source electrical partition.
    spines : sequence of Spine
        Explicit additions with physical dimensions and source attachments.

    Returns
    -------
    H01SpineAssembly
        Geometry and remapped selections, with original and added identities.
    """
    additions = tuple(spines)
    morphology, mapping, heads = add_spines(imported.morphology, additions)
    intervals = {key: region.evaluate(imported.morphology).intervals for key, region in regions.items()}
    partition = remap_electrical_regions(mapping, intervals, additions)
    names = {view.name: index for index, view in enumerate(morphology.branches)}
    signature = _geometry_signature(morphology)
    mapped = {key: _Region(signature, tuple((names[name], lo, hi) for name, lo, hi in rows),
                           'explicit spine parent inheritance') for key, rows in partition.items()}
    source_soma = imported.anatomy().soma_location().evaluate(imported.morphology).points[0]
    soma_name, soma_x = remap_location(mapping, *source_soma)
    # Numeric coordinates make records JSON-serializable and independent of
    # branch names; the source map is stored separately from original evidence.
    numeric = {str(index): [(lo, hi, names[name]) for lo, hi, name in rows]
               for index, rows in mapping.items()}
    evidence = dict(source_sha256=imported.source_sha256, geometry_sha256=signature,
        source_map=numeric, heads={key: names[name] for key, name in heads.items()},
        additions=[asdict(spine) for spine in additions], source_soma=list(source_soma),
        assembled_soma=[names[soma_name], soma_x],
        geometry_basis='Explicit cylinders extending from the source centreline; no measured surface claim',
        electrical_basis='Each spine inherits its source attachment electrical family')
    return H01SpineAssembly(morphology, mapped, AtLocation(names[soma_name], soma_x), evidence)


def assembled_location(record, site):
    """Map a source coordinate using an instance construction record.

    Parameters
    ----------
    record : dict
        Cell record with optional spine_assembly evidence.
    site : sequence
        Original branch index and normalized cable position.

    Returns
    -------
    tuple
        Assembled branch index and position; unchanged for a baseline instance.
    """
    if 'spine_assembly' not in record:
        return tuple(site)
    mapping = {int(key): value for key, value in record['spine_assembly']['source_map'].items()}
    return remap_location(mapping, *site)
