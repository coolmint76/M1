//***** CANON MEDICAL SYSTEMS CORPORATION ** COMPANY CONFIDENTIAL *****
///////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2013 Canon Medical Systems Corporation, All Rights Reserved
//  Created: 02-Dec-2013 Wheaton TMRU
///////////////////////////////////////////////////////////////////////////////
#include <MR/common/stlpdefs.h>
#include <MR/libpattern/PatternTrajectoryCal.h>
#include <MR/libpattern/MomentTable.h>

void PatternTrajectoryCal::EpvFillPattern()
{
    m_moment->setEncodeAxes(m_axis);
    m_moment->setPattern(this);
    initTable(m_dim->getLoopList());

    LoopSet lps = m_dim->getLoopList();

    int32_t numEncode = lps.encodeDim();

    //int32_t kyPeak = getKyPeak();

    // adjust step size so it goes through zero
    flt64_t trajectoryStep;
    if (numEncode > 2)
    {
        trajectoryStep = (numEncode % 2) ? (2.0 / (flt64_t)(numEncode - 1)) : (2.0 / (flt64_t)numEncode);
    }
    else if (numEncode == 2)
    {
        trajectoryStep = 2.0; // produce only -1 and +1 encodes
    }
    else
    {
        trajectoryStep = 0;
    }

    lps.reset();
    do
    {
        if (lps.isCond())
        {
            (*m_EncodeTable)[lps].attrib                = VFC_SRTTBL_ATTRB_CONDITION;
            (*m_EncodeTable)[lps].kIndex                = 0;
            (*m_EncodeTable)[lps].phaseEncode[m_axis]   = 0;
            (*m_EncodeTable)[lps].freqEncode            = 0;
            (*m_EncodeTable)[lps].acqIndex              = ACQINDEX_NOACQ;
        }
        else if (lps.isNoise())
        {
            (*m_EncodeTable)[lps].attrib                = VFC_SRTTBL_ATTRB_NOISE;
            (*m_EncodeTable)[lps].kIndex                = lps.encodeOffset();
            (*m_EncodeTable)[lps].phaseEncode[m_axis]   = 0;
            (*m_EncodeTable)[lps].freqEncode            = 0;
            (*m_EncodeTable)[lps].acqIndex              = lps.reconOffset();
        }
        else if (lps.isTempl())
        {
            (*m_EncodeTable)[lps].attrib                = VFC_SRTTBL_ATTRB_TEMPLATE;
            (*m_EncodeTable)[lps].kIndex                = lps.encodeOffset();
            (*m_EncodeTable)[lps].phaseEncode[m_axis]   = 0;
            (*m_EncodeTable)[lps].freqEncode            = 0;
            (*m_EncodeTable)[lps].acqIndex              = lps.reconOffset();
        }
        else
        {
            int32_t kIndex = lps.encodeOffset();
            // simple linear ramp on SS direction (hitting zero)
            (*m_EncodeTable)[lps].freqEncode[ax_x]  = 0;
            (*m_EncodeTable)[lps].freqEncode[ax_y]  = 0;
            (*m_EncodeTable)[lps].freqEncode[ax_z]  = -1.0 + trajectoryStep * kIndex;

            (*m_EncodeTable)[lps].attrib                = VFC_SRTTBL_ATTRB_NONE;
            (*m_EncodeTable)[lps].kIndex                = kIndex;
            (*m_EncodeTable)[lps].phaseEncode[m_axis]   = 0;
            (*m_EncodeTable)[lps].acqIndex              = lps.reconOffset();
        }
    }
    while (lps.next());

    // Update moment index table in Encode object
    Table<F64Vec3> phaseEncodeTbl("peenc", F64Vec3());
    createPhaseEncodeTable(phaseEncodeTbl);
    m_moment->setIndexTable(phaseEncodeTbl);

    Table<F64Vec3> freqEncodeTbl("feenc", F64Vec3());
    createFreqEncodeTable(freqEncodeTbl);
    m_moment->setRotationTable(freqEncodeTbl);
}

void PatternTrajectoryCal::generateSortTable(PhaseEncodeSort &sortTable)
{
    const Table<EncodeData>* data = getTable();

    int totalSize = m_dim->getFull() + m_dim->extra();

    sortTable.generateSortTable(data, totalSize);
}

void PatternTrajectoryCal::EpvSaveParams(SeqDBSave& par)
{
    generateSortTable(m_pesort);
    m_pesort.saveParams(par.PeSort);
}