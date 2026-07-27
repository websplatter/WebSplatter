import { INRIAV1PlyParser } from './INRIAV1PlyParser.ts';
import { PlyParserUtils } from './PlyParserUtils.ts';
import { PlyFormat } from './PlyFormat.ts';

export class PlyParser {

    static parseToUncompressedSplatArray(plyBuffer: ArrayBuffer, outSphericalHarmonicsDegree = 0) {
        const plyFormat = PlyParserUtils.determineHeaderFormatFromPlyBuffer(plyBuffer);
        if (plyFormat === PlyFormat.INRIAV1) {
            return INRIAV1PlyParser.parseToUncompressedSplatArray(plyBuffer, outSphericalHarmonicsDegree);
        } else if (plyFormat === PlyFormat.INRIAV2) {
             // TODO: Implement!
            throw new Error('parseToUncompressedSplatArray() is not implemented for INRIA V2 PLY files');
        }
    }

}
