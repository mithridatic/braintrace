"""Immutable anatomical provenance and explicitly synthetic H01 evolution."""

from dataclasses import dataclass
import json
import math


def _site(value):
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and type(value[0]) is int and value[0] >= 0
            and isinstance(value[1], (int, float)) and math.isfinite(value[1]) and 0 <= value[1] <= 1)


def _validate(doc):
    if doc.get('schema') != 'h01-topology-v1':
        raise ValueError('Unsupported H01 topology schema')
    sources, instances, contacts = (doc[name] for name in ('sources', 'instances', 'contacts'))
    active, edges = doc['active_cells'], doc['active_contacts']
    if not active or len(active) != len(set(active)) or len(edges) != len(set(edges)):
        raise ValueError('Active topology must be nonempty and have unique identities')
    if not set(active) <= instances.keys() or not set(edges) <= contacts.keys():
        raise ValueError('Active topology references missing historical records')
    if type(doc['next_id']) is not int or doc['next_id'] < 0:
        raise ValueError('Synthetic identity counter must be a nonnegative integer')
    for identity, source in sources.items():
        if (not isinstance(identity, str) or not identity or source['polarity'] not in ('E', 'I')
                or type(source['component']) is not int or source['component'] < 0
                or not _site(source['soma_site'])
                or source['output_site'] is not None and not _site(source['output_site'])):
            raise ValueError('Invalid source identity, component, polarity or cable site')
        digest = source['source_sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Source morphology requires a SHA256 digest')
        if source['profile'].get('polarity', source['polarity']) != source['polarity']:
            raise ValueError('Source profile cannot change E/I identity')
    for identity, instance in instances.items():
        if not isinstance(identity, str) or instance['source_id'] not in sources:
            raise ValueError('Instances require string IDs and a known immutable source')
        parent = instance['parent_id']
        if instance['synthetic'] != (parent is not None):
            raise ValueError('Synthetic instances must retain their parent identity')
        seen = {identity}
        while parent is not None:
            if parent in seen or parent not in instances:
                raise ValueError('Instance lineage is cyclic or missing its parent')
            seen.add(parent)
            if instances[parent]['source_id'] != instance['source_id']:
                raise ValueError('Clones must inherit their source and E/I identity')
            parent = instances[parent]['parent_id']
    pairs = set()
    for identity, contact in contacts.items():
        pre, post = contact['pre'], contact['post']
        if pre not in instances or post not in instances or not _site(contact['pre_site']) or not _site(contact['post_site']):
            raise ValueError('Contact has an unknown endpoint or invalid cable placement')
        polarity = sources[instances[pre]['source_id']]['polarity']
        reversal, tau = (0., 2.) if polarity == 'E' else (-80., 5.)
        if (contact['reversal_mv'], contact['tau_ms'], contact['delay_ms']) != (reversal, tau, .5):
            raise ValueError('Contact receptor settings must preserve source identity and clocks')
        if not math.isfinite(contact['initial_weight_us']) or contact['initial_weight_us'] < 0:
            raise ValueError('Contact magnitudes must be finite and nonnegative')
        if identity in edges:
            if pre not in active or post not in active or (pre, post) in pairs:
                raise ValueError('Active contacts require active endpoints and unique directed pairs')
            pairs.add((pre, post))


@dataclass(frozen=True)
class H01Topology:
    """Store an immutable topology document with permanent lineage records.

    Parameters
    ----------
    payload : str
        Validated JSON document; normally construct with :meth:`from_dict`.

    Notes
    -----
    Activity is separate from historical records. Every mutation returns a new
    topology and retains all source, instance and contact provenance.
    """

    payload: str

    def __post_init__(self):
        _validate(json.loads(self.payload))

    @classmethod
    def from_dict(cls, document):
        """Validate and detach a topology document from caller-owned mappings.

        Parameters
        ----------
        document : dict
            Source profiles, lineage, contacts and active identities.

        Returns
        -------
        H01Topology
            Immutable canonical snapshot.
        """
        return cls(json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False))

    def to_dict(self):
        """Return an independent mutable copy for serialization or inspection.

        Returns
        -------
        dict
            Complete topology and historical provenance.
        """
        return json.loads(self.payload)

    def _finish(self, doc, operation, stage, **details):
        if not isinstance(stage, str) or not stage:
            raise ValueError('Mutation requires a durable stage identity')
        doc['mutations'].append(dict(operation=operation, stage=stage, **details))
        return self.from_dict(doc)

    @staticmethod
    def _identity(doc, kind):
        identity = f'synthetic-{kind}-{doc["next_id"]}'
        doc['next_id'] += 1
        if identity in doc['instances'] or identity in doc['contacts']:
            raise ValueError('Synthetic identity counter would reuse historical provenance')
        return identity

    def clone(self, parent, *, stage):
        """Clone an active cell and its incident contacts with synthetic labels.

        Parameters
        ----------
        parent : str
            Active donor instance ID.
        stage : str
            Coordinator stage identity.

        Returns
        -------
        H01Topology
            Child snapshot; the final active cell is the new clone.
        """
        doc = self.to_dict()
        if parent not in doc['active_cells']:
            raise ValueError('Cannot clone an inactive instance')
        child = self._identity(doc, 'cell')
        doc['instances'][child] = dict(source_id=doc['instances'][parent]['source_id'],
                                      parent_id=parent, synthetic=True, created_stage=stage)
        doc['active_cells'].append(child)
        for identity in tuple(doc['active_contacts']):
            edge = doc['contacts'][identity]
            if parent not in (edge['pre'], edge['post']):
                continue
            copied = dict(edge, pre=child if edge['pre'] == parent else edge['pre'],
                          post=child if edge['post'] == parent else edge['post'],
                          synthetic=True, parent_id=identity, created_stage=stage)
            name = self._identity(doc, 'contact')
            doc['contacts'][name] = copied
            doc['active_contacts'].append(name)
        return self._finish(doc, 'clone', stage, parent=parent, child=child)

    def prune_cell(self, identity, *, stage):
        """Deactivate a cell and incident contacts while retaining their records.

        Parameters
        ----------
        identity : str
            Active original or synthetic instance.
        stage : str
            Coordinator stage identity.

        Returns
        -------
        H01Topology
            Nonempty surviving active topology.
        """
        doc = self.to_dict()
        if identity not in doc['active_cells']:
            raise ValueError('Cannot prune an inactive instance')
        if len(doc['active_cells']) == 1:
            raise ValueError('Cannot create an empty active network')
        doc['active_cells'].remove(identity)
        doc['active_contacts'] = [key for key in doc['active_contacts']
            if identity not in (doc['contacts'][key]['pre'], doc['contacts'][key]['post'])]
        return self._finish(doc, 'prune_cell', stage, instance=identity)

    def add_contact(self, pre, post, *, stage, weight_us=.01):
        """Add an explicitly synthetic directed contact using frozen cable sites.

        Parameters
        ----------
        pre, post : str
            Distinct active source and target instances with no active contact.
        stage : str
            Coordinator stage identity.
        weight_us : float, optional
            Initial nonnegative conductance magnitude.

        Returns
        -------
        H01Topology
            Snapshot with a designated-output-to-soma synthetic contact.
        """
        doc = self.to_dict()
        pairs = {(doc['contacts'][key]['pre'], doc['contacts'][key]['post']) for key in doc['active_contacts']}
        if pre == post or pre not in doc['active_cells'] or post not in doc['active_cells'] or (pre, post) in pairs:
            raise ValueError('Synthetic contact requires an absent directed pair of active distinct cells')
        source, target = (doc['sources'][doc['instances'][identity]['source_id']] for identity in (pre, post))
        reversal, tau = (0., 2.) if source['polarity'] == 'E' else (-80., 5.)
        identity = self._identity(doc, 'contact')
        doc['contacts'][identity] = dict(pre=pre, post=post, pre_site=source['output_site'] or source['soma_site'],
            post_site=target['soma_site'], synthetic=True, parent_id=None, annotation_id=None,
            created_stage=stage, initial_weight_us=weight_us, reversal_mv=reversal, tau_ms=tau, delay_ms=.5)
        doc['active_contacts'].append(identity)
        return self._finish(doc, 'add_contact', stage, contact=identity)

    def prune_contact(self, identity, *, stage):
        """Deactivate a contact without removing its historical record.

        Parameters
        ----------
        identity : str
            Active contact ID.
        stage : str
            Coordinator stage identity.

        Returns
        -------
        H01Topology
            Snapshot with the contact deactivated.
        """
        doc = self.to_dict()
        if identity not in doc['active_contacts']:
            raise ValueError('Cannot prune an inactive contact')
        doc['active_contacts'].remove(identity)
        return self._finish(doc, 'prune_contact', stage, contact=identity)

    def rank_absent_pairs(self, activity, gradients):
        """Rank absent pairs deterministically by gradient then activity magnitude.

        Parameters
        ----------
        activity : mapping
            Per-instance measured activity.
        gradients : mapping
            Directed-pair gradient scores.

        Returns
        -------
        list of tuples
            Ranked source/target IDs, using stable strings to break ties.
        """
        doc = self.to_dict()
        occupied = {(doc['contacts'][key]['pre'], doc['contacts'][key]['post']) for key in doc['active_contacts']}
        candidates = [(a, b) for a in doc['active_cells'] for b in doc['active_cells'] if a != b and (a, b) not in occupied]
        def rank(pair):
            gradient = abs(float(gradients.get(pair, 0.)))
            score = abs(float(activity.get(pair[0], 0.))*float(activity.get(pair[1], 0.)))
            if not math.isfinite(gradient) or not math.isfinite(score):
                raise ValueError('Contact ranking requires finite measured scores')
            return -gradient, -score, *pair
        return sorted(candidates, key=rank)
