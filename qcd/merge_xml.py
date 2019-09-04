from typing import Optional
from xml.dom import minidom
from xml.etree import ElementTree

import plac


@plac.annotations(
    files=plac.Annotation('The list of annotated XML files to merge.', type=str),
    output_path=plac.Annotation('Where to save the output file. '
                                'If set to None, the merged file is printed to stdout.',
                                type=str, kind='option'),
)
def main(output_path: Optional[str] = None, *files):
    """Merge annotations from multiple XML files into a single XML file."""

    if len(files) > 0:
        tree = ElementTree.ElementTree(element=ElementTree.Element('document'))

        document_sections = [ElementTree.parse(file).getroot().findall('section') for file in files]

        for i, sections in enumerate(zip(*document_sections)):
            for section in sections[1:]:
                annotations = sections[0].find('annotations')

                for annotation in section.find('annotations'):
                    annotations.append(annotation)

            tree.getroot().append(sections[0])

        tree_string = ElementTree.tostring(tree.getroot(), encoding='unicode')

        # Use minidom so we get access to pretty printing methods.
        xml_dom = minidom.parseString(tree_string)

        if output_path:
            with open(output_path, 'w') as f:
                xml_dom.writexml(f, addindent='    ', newl='\n')
        else:
            print(xml_dom.toprettyxml(indent='    '))


if __name__ == '__main__':
    plac.call(main)
