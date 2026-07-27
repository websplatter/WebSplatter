import { clamp } from '../../util.ts';
import { HeaderType, PlyParserUtils } from './PlyParserUtils.ts';
import { UncompressedSplatArray } from '../UncompressedSplatArray.ts';

const BaseFieldNamesToRead = ['scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3', 'x', 'y', 'z',
                              'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'red', 'green', 'blue', 'f_rest_0'];

const BaseFieldsToReadIndexes = BaseFieldNamesToRead.map((e, i) => i);

const [
        SCALE_0, SCALE_1, SCALE_2, ROT_0, ROT_1, ROT_2, ROT_3, X, Y, Z, F_DC_0, F_DC_1, F_DC_2, OPACITY, RED, GREEN, BLUE, F_REST_0
      ] = BaseFieldsToReadIndexes;

export class INRIAV1PlyParser {

    static decodeHeaderLines(headerLines: string[]): HeaderType {

        // Find max f_rest_* index present in header lines
        let maxFrestIndex = 0;
        for (const line of headerLines) {
            if (line.includes('f_rest_')) {
                const m = line.match(/f_rest_(\d+)/);
                if (m) {
                    const idx = parseInt(m[1], 10);
                    if (!Number.isNaN(idx)) maxFrestIndex = Math.max(maxFrestIndex, idx);
                }
            }
        }

        const shRemainingFieldNamesToRead: string[] = [];
        for (let i = 1; i <= maxFrestIndex; i++) shRemainingFieldNamesToRead.push(`f_rest_${i}`);

        const fieldNamesToRead = [...BaseFieldNamesToRead, ...shRemainingFieldNamesToRead];
        const fieldsToReadIndexes = fieldNamesToRead.map((e, i) => i);

        const fieldNameIdMap: Record<string, number> = fieldsToReadIndexes.reduce((acc, element) => {
            acc[fieldNamesToRead[element]] = element;
            return acc;
        }, {});
        const header = PlyParserUtils.decodeSectionHeader(headerLines, fieldNameIdMap, 0);
        header.fieldsToReadIndexes = fieldsToReadIndexes;
        return header;
    }

    static decodeHeaderText(headerText: string) {
        const headerLines = PlyParserUtils.convertHeaderTextToLines(headerText);
        const header: HeaderType = INRIAV1PlyParser.decodeHeaderLines(headerLines);
        header.headerText = headerText;
        header.headerSizeBytes = headerText.indexOf(PlyParserUtils.HeaderEndToken) + PlyParserUtils.HeaderEndToken.length + 1;
        return header;
    }

    static decodeHeaderFromBuffer(plyBuffer: ArrayBuffer): HeaderType {
        const headerText = PlyParserUtils.readHeaderFromBuffer(plyBuffer);
        return INRIAV1PlyParser.decodeHeaderText(headerText);
    }

    static findSplatData(plyBuffer: ArrayBuffer, header: { headerLines?: string[]; headerStartLine?: number; headerEndLine?: number; fieldTypes?: number[]; fieldIds?: number[]; fieldOffsets?: number[]; bytesPerVertex?: number; vertexCount?: number; dataSizeBytes?: number; endOfHeader?: boolean; sectionName?: any; sphericalHarmonicsDegree?: number; sphericalHarmonicsCoefficientsPerChannel?: number; sphericalHarmonicsDegreeFields?: number[][]; headerSizeBytes?: any; }) {
        return new DataView(plyBuffer, header.headerSizeBytes);
    }

    static parseToUncompressedSplatArraySection(header: { sphericalHarmonicsDegree: number; }, fromSplat: number, toSplat: number, splatData: DataView, splatDataOffset: number,
                                         splatArray: { addSplat: (arg0: number[]) => void; }, outSphericalHarmonicsDegree = 0) {
        outSphericalHarmonicsDegree = Math.min(outSphericalHarmonicsDegree, header.sphericalHarmonicsDegree);
        for (let i = fromSplat; i <= toSplat; i++) {
            const parsedSplat = INRIAV1PlyParser.parseToUncompressedSplat(splatData, i, header,
                                                                          splatDataOffset, outSphericalHarmonicsDegree);
            splatArray.addSplat(parsedSplat);
        }
    }

    static decodeSectionSplatData(sectionSplatData: DataView<any>, splatCount: number, sectionHeader: { headerLines?: any[]; headerStartLine?: number; headerEndLine?: number; fieldTypes?: any[]; fieldIds?: any[]; fieldOffsets?: any[]; bytesPerVertex?: number; vertexCount?: number; dataSizeBytes?: number; endOfHeader?: boolean; sectionName?: any; sphericalHarmonicsDegree: number; sphericalHarmonicsCoefficientsPerChannel?: number; sphericalHarmonicsDegreeFields?: number[][]; }, outSphericalHarmonicsDegree: number) {
        outSphericalHarmonicsDegree = Math.min(outSphericalHarmonicsDegree, sectionHeader.sphericalHarmonicsDegree);
        const splatArray = new UncompressedSplatArray(outSphericalHarmonicsDegree);
        for (let row = 0; row < splatCount; row++) {
            const newSplat = INRIAV1PlyParser.parseToUncompressedSplat(sectionSplatData, row, sectionHeader,
                                                                        0, outSphericalHarmonicsDegree);
            splatArray.addSplat(newSplat);
        }
        return splatArray;

    }

    static parseToUncompressedSplat = function() {

        let rawSplat = [];
        // const tempRotation = new THREE.Quaternion();

        const OFFSET_X = UncompressedSplatArray.OFFSET.X;
        const OFFSET_Y = UncompressedSplatArray.OFFSET.Y;
        const OFFSET_Z = UncompressedSplatArray.OFFSET.Z;

        const OFFSET_SCALE0 = UncompressedSplatArray.OFFSET.SCALE0;
        const OFFSET_SCALE1 = UncompressedSplatArray.OFFSET.SCALE1;
        const OFFSET_SCALE2 = UncompressedSplatArray.OFFSET.SCALE2;

        const OFFSET_ROTATION0 = UncompressedSplatArray.OFFSET.ROTATION0;
        const OFFSET_ROTATION1 = UncompressedSplatArray.OFFSET.ROTATION1;
        const OFFSET_ROTATION2 = UncompressedSplatArray.OFFSET.ROTATION2;
        const OFFSET_ROTATION3 = UncompressedSplatArray.OFFSET.ROTATION3;

        const OFFSET_FDC0 = UncompressedSplatArray.OFFSET.FDC0;
        const OFFSET_FDC1 = UncompressedSplatArray.OFFSET.FDC1;
        const OFFSET_FDC2 = UncompressedSplatArray.OFFSET.FDC2;
        const OFFSET_OPACITY = UncompressedSplatArray.OFFSET.OPACITY;

        function OFFSET_FRC(i: number) {
            return UncompressedSplatArray.OFFSET.FRC0 + i;
        }

        return function(splatData: DataView, row: any, header, splatDataOffset = 0, outSphericalHarmonicsDegree = 0) {
            outSphericalHarmonicsDegree = Math.min(outSphericalHarmonicsDegree, header.sphericalHarmonicsDegree);
            INRIAV1PlyParser.readSplat(splatData, header, row, splatDataOffset, rawSplat);
            const newSplat = UncompressedSplatArray.createSplat(outSphericalHarmonicsDegree);
            if (rawSplat[SCALE_0] !== undefined) {
                newSplat[OFFSET_SCALE0] = Math.exp(rawSplat[SCALE_0]);
                newSplat[OFFSET_SCALE1] = Math.exp(rawSplat[SCALE_1]);
                newSplat[OFFSET_SCALE2] = Math.exp(rawSplat[SCALE_2]);
            } else {
                newSplat[OFFSET_SCALE0] = 0.01;
                newSplat[OFFSET_SCALE1] = 0.01;
                newSplat[OFFSET_SCALE2] = 0.01;
            }

            if (rawSplat[F_DC_0] !== undefined) {
                const SH_C0 = 0.28209479177387814;
                newSplat[OFFSET_FDC0] = (0.5 + SH_C0 * rawSplat[F_DC_0]) * 255;
                newSplat[OFFSET_FDC1] = (0.5 + SH_C0 * rawSplat[F_DC_1]) * 255;
                newSplat[OFFSET_FDC2] = (0.5 + SH_C0 * rawSplat[F_DC_2]) * 255;
            } else if (rawSplat[RED] !== undefined) {
                newSplat[OFFSET_FDC0] = rawSplat[RED] * 255;
                newSplat[OFFSET_FDC1] = rawSplat[GREEN] * 255;
                newSplat[OFFSET_FDC2] = rawSplat[BLUE] * 255;
            } else {
                newSplat[OFFSET_FDC0] = 0;
                newSplat[OFFSET_FDC1] = 0;
                newSplat[OFFSET_FDC2] = 0;
            }

            if (rawSplat[OPACITY] !== undefined) {
                newSplat[OFFSET_OPACITY] = (1 / (1 + Math.exp(-rawSplat[OPACITY]))) * 255;
            }

            newSplat[OFFSET_FDC0] = clamp(Math.floor(newSplat[OFFSET_FDC0]), 0, 255);
            newSplat[OFFSET_FDC1] = clamp(Math.floor(newSplat[OFFSET_FDC1]), 0, 255);
            newSplat[OFFSET_FDC2] = clamp(Math.floor(newSplat[OFFSET_FDC2]), 0, 255);
            newSplat[OFFSET_OPACITY] = clamp(Math.floor(newSplat[OFFSET_OPACITY]), 0, 255);

            if (outSphericalHarmonicsDegree >= 1 && rawSplat[F_REST_0] !== undefined) {
                let dst = 0;
                for (let d = 1; d <= outSphericalHarmonicsDegree; d++) {
                    const degreeFields = header.sphericalHarmonicsDegreeFields?.[d];
                    if (!degreeFields) {
                        throw new Error(`Missing spherical harmonics degree fields for degree ${d}`);
                    }
                    for (let j = 0; j < degreeFields.length; j++) {
                        newSplat[OFFSET_FRC(dst++)] = rawSplat[degreeFields[j]];
                    }
                }
            }

            const scale = Math.sqrt(rawSplat[ROT_0] * rawSplat[ROT_0] + rawSplat[ROT_1] * rawSplat[ROT_1] + rawSplat[ROT_2] * rawSplat[ROT_2] + rawSplat[ROT_3] * rawSplat[ROT_3]);

            newSplat[OFFSET_ROTATION0] = rawSplat[ROT_0] / scale;
            newSplat[OFFSET_ROTATION1] = rawSplat[ROT_1] / scale;
            newSplat[OFFSET_ROTATION2] = rawSplat[ROT_2] / scale;
            newSplat[OFFSET_ROTATION3] = rawSplat[ROT_3] / scale;

            newSplat[OFFSET_X] = rawSplat[X];
            newSplat[OFFSET_Y] = rawSplat[Y];
            newSplat[OFFSET_Z] = rawSplat[Z];

            return newSplat;
        };

    }();

    static readSplat(splatData: DataView, header, row: number, dataOffset: number, rawSplat: any[]) {
        return PlyParserUtils.readVertex(splatData, header, row, dataOffset, header.fieldsToReadIndexes, rawSplat, true);
    }

    static parseToUncompressedSplatArray(plyBuffer: ArrayBuffer, outSphericalHarmonicsDegree = 0) {
        const { header, splatCount, splatData } = separatePlyHeaderAndData(plyBuffer);
        return INRIAV1PlyParser.decodeSectionSplatData(splatData, splatCount, header, outSphericalHarmonicsDegree);
    }
}

function separatePlyHeaderAndData(plyBuffer: ArrayBuffer) {
    const header: HeaderType = INRIAV1PlyParser.decodeHeaderFromBuffer(plyBuffer);
    const splatCount = header.vertexCount;
    const splatData = INRIAV1PlyParser.findSplatData(plyBuffer, header);
    return {
        header,
        splatCount,
        splatData
    };
}
