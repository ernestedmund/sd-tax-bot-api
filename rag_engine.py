"""
County Property Tax Assistant - RAG Engine
==========================================
Chunks the knowledge base, builds a TF-IDF vector index (no external
embedding API needed), retrieves relevant chunks at query time, and
calls Claude Haiku with only those chunks.

For production: swap TF-IDF for a real embedding model (see comments).
"""

import json
import re
import math
import os
from collections import Counter
from anthropic import Anthropic

RAW_KNOWLEDGE_BASE = """
## P13-001 | Prop 13 Basics | What is Proposition 13?
Proposition 13 is a California law passed by voters in 1978 that limits how much your property's taxable value can increase each year. When you buy a property or complete new construction, a "base year value" is set — and that value can only go up by a maximum of 2% per year, no matter what the real estate market does.
Source: BOE Publication 29 · P13-001

## P13-002 | Prop 13 Basics | What is a base year value?
Your base year value is the taxable value assigned to your property when it is purchased or when new construction is completed. For a purchase, it is generally equal to the sale price — provided the sale was an arm's-length transaction between an unrelated buyer and seller. The Assessor validates each sale; if the sale price reflects distressed conditions (foreclosure, short sale, auction, or a purchase from a family member or friend), the enrolled value may differ from what you paid. This becomes the starting point for your property taxes, and it can only increase by up to 2% per year going forward.
Source: BOE Publication 29 · P13-002

## P13-003 | Prop 13 Basics | What is the property tax rate in California?
The base property tax rate in California is 1% of your property's assessed value. However, your total tax bill is typically higher than 1% because it also includes voter-approved charges such as school bonds, Mello-Roos special taxes, and other local assessments that vary by location.
Source: BOE Publication 29 · P13-003

## P13-004 | Prop 13 Basics | Why did my assessed value go up more than 2%?
There are three reasons your assessed value could increase by more than 2%: (1) A Change in Ownership occurred, which triggers reassessment to current market value; (2) new construction was completed, which adds the market value of the new work; or (3) your property was previously under a Proposition 8 decline-in-value reduction, and as the market recovers, the assessed value can increase by more than 2% per year until it returns to the Prop 13 factored base year value. If none of these apply, contact your county Assessor's office to review your account.
Source: BOE Publication 29 · P13-004

## P13-005 | Prop 13 Basics | What triggers a reassessment to full market value?
Two events trigger a reassessment: a Change in Ownership (such as a sale) and the completion of new construction. Outside of these events, assessed value can only increase by up to 2% per year.
Source: BOE Publication 29 · P13-005

## P13-006 | Prop 13 Basics | Can my assessed value go down? What is Prop 8?
Yes. If current market value falls below your Prop 13 assessed value, the county must temporarily reduce your assessed value. This is called a Proposition 8 reduction. When the market recovers, your value is restored up to the Prop 13 factored base year value — never above it.
Source: BOE Publication 9 · P13-006

## P13-007 | Prop 13 Basics | Difference between assessed value and market value?
Market value is what your property would sell for today. Assessed value is what the county uses to calculate taxes. Because of the 2% annual cap, assessed value may be significantly lower than market value, especially for long-term owners.
Source: BOE Publication 29 · P13-007

## P13-008 | Prop 13 Basics | I just bought a home. How will taxes be calculated?
Your purchase price is used to establish your new base year value, provided it reflects an arm's-length sale between an unrelated buyer and seller. The Assessor validates the transaction — in a typical open-market sale the purchase price is enrolled as-is, but foreclosures, short sales, auctions, or purchases from family may be assessed at an independently determined market value instead. Annual tax is approximately 1% of the base year value plus local charges, and it can only increase by up to 2% per year going forward unless you make significant improvements or the property changes ownership again.
Source: BOE Publication 29 · P13-008

## CIO-001 | Change in Ownership | What is a Change in Ownership?
A Change in Ownership (CIO) is a transfer of property that causes the county to reassess it at current market value. The new market value becomes the new base year value, subject to Prop 13 rules going forward. This can significantly increase taxes if market value has risen. Note: the Assessor determines market value independently — in a standard arm's-length sale the purchase price is typically enrolled, but distressed sales, foreclosures, or non-arm's-length transfers may be assessed at a value the Assessor determines separately.
Source: BOE Publication 150-C · CIO-001

## CIO-002 | Change in Ownership | What events trigger a Change in Ownership?
Common triggers: sale or purchase of property, a gift of real property, transferring into or out of certain trusts, transferring into or out of an LLC or corporation, and inheriting property in some circumstances. Any transfer of beneficial ownership — regardless of whether money changes hands — can trigger reassessment.
Source: BOE Publication 150-C · CIO-002

## CIO-003 | Change in Ownership | What does NOT trigger a Change in Ownership?
Exclusions include: transfers between spouses, transfers between registered domestic partners, certain parent-child transfers, certain grandparent-grandchild transfers, transfers into a revocable living trust where the transferor remains the beneficiary, and transfers that only change how title is held without changing actual ownership.
Source: BOE Publication 150-C · CIO-003

## CIO-004 | Change in Ownership | What is the Parent-Child Exclusion?
Under Proposition 19 (effective February 16, 2021), the exclusion applies only to a primary residence, and only if the child uses it as their primary residence within one year of the transfer. The prior exclusion for rental or vacation properties was largely eliminated. File form BOE-58-AH to claim.
Source: BOE Publication 150-C · CIO-004

## CIO-005 | Change in Ownership | What is the Grandparent-Grandchild Exclusion?
Under Prop 19: applies only to a primary residence, only if all of the grandchild's parents who are children of the grandparent are deceased, and only if the grandchild moves in within one year. File form BOE-58-AH.
Source: BOE Publication 150-C · CIO-005

## CIO-006 | Change in Ownership | Does transferring to my spouse trigger reassessment?
No. Transfers between spouses are excluded, including transfers during marriage, as a result of divorce, or upon death. Registered domestic partners receive the same exclusion.
Source: BOE Publication 150-C · CIO-006

## CIO-007 | Change in Ownership | I inherited a property. Will it be reassessed?
It depends. Inheriting a property can trigger reassessment, but exclusions may apply. Under Prop 19, inheriting a primary residence from a parent or grandparent and using it as your primary residence within one year may qualify for an exclusion. Rental and investment properties are generally reassessed.
Source: BOE Publication 150-C · CIO-007

## CIO-008 | Change in Ownership | Does transferring into a trust trigger reassessment?
Transferring into a revocable living trust where you remain the beneficiary does not trigger reassessment. Transferring into an irrevocable trust or one where someone else becomes the beneficial owner may trigger a CIO.
Source: BOE Publication 150-C · CIO-008

## CIO-010 | Change in Ownership | What is a Preliminary Change of Ownership Report (PCOR)?
A PCOR (form BOE-502-A) must be filed with the county recorder whenever real property is transferred. It helps the Assessor determine whether reassessment applies or an exclusion exists. Failure to file may result in a $20 fee.
Source: BOE Publication 4 · CIO-010

## CIO-011 | Change in Ownership | Will my foreclosure or distressed sale price be my new assessed value?
Not necessarily. While purchase price is generally presumed to equal market value in an arm's-length sale, the Assessor must validate each transaction. For foreclosures, short sales, auctions, REO purchases, and purchases from family members or friends, the sale price may not reflect open-market conditions — one or more elements of a fair market value transaction may be missing. In those cases the Assessor may enroll a value higher or lower than what you paid, based on an independent market analysis. If you believe the enrolled value is incorrect, you can request an informal review or file a formal appeal with your county Assessor's office.
Source: BOE Property Tax Rule 2 · CIO-011

## RC-001 | Roll Corrections | There is an error on my assessment — can it be fixed?
Yes. If there is a factual error on your assessment roll — such as a wrong property size, incorrect owner name, or a clerical mistake — the Assessor can correct it after the roll has been delivered to the Auditor, generally within four years of the assessment being made. Errors involving a decline in value that was not properly reflected must be corrected within one year. If a correction increases your taxes, the Assessor must notify you and explain the review process. If it decreases your taxes, the Board of Supervisors must consent. To report an error, contact your county Assessor's office.
Source: BOE Property Tax Rule 263 · RC-001

## NC-001 | New Construction | What counts as new construction for property tax purposes?
New construction includes any addition of a new structure, expansion of an existing structure, and any conversion requiring a building permit. Only the newly constructed portion is reassessed — the rest retains its existing base year value.
Source: BOE Publication 150-D · NC-001

## NC-002 | New Construction | Do repairs and maintenance trigger reassessment?
No. Routine repairs such as replacing a roof with the same materials, repainting, or repairing plumbing do not trigger reassessment. Reassessment requires that work adds value, extends useful life beyond original condition, or converts to a new use.
Source: BOE Publication 150-D · NC-002

## NC-004 | New Construction | I added a room. How will my taxes be affected?
Adding a room triggers reassessment but only for the new addition. The Assessor determines the market value of the added square footage as of completion date. Your existing home retains its current base year value.
Source: BOE Publication 150-D · NC-004

## NC-007 | New Construction | I built an ADU. Will my taxes go up?
Yes, in most cases. A newly constructed ADU is new construction and will be assessed at market value when substantially completed. Only the ADU is reassessed; your existing home and land retain their current assessed values.
Source: BOE Publication 150-D · NC-007

## NC-008 | New Construction | Are solar panels considered new construction?
Active solar energy systems are currently excluded from new construction reassessment under California law (R&T Code section 73). This exclusion expires January 1, 2027, unless extended by the legislature.
Source: BOE Publication 150-D · NC-008

## NC-010 | New Construction | Are accessibility improvements excluded?
Yes. Improvements made to accommodate a disabled person — including ramps, widened doorways, and modified bathrooms — are excluded from new construction reassessment.
Source: BOE Publication 150-D · NC-010

## SUP-001 | Supplemental Assessments | What is a supplemental assessment?
A supplemental assessment is issued when a property's taxable value increases mid-year due to a Change in Ownership or new construction. It captures the value increase between the event date and fiscal year end, resulting in a separate supplemental tax bill.
Source: BOE Publication 26 · SUP-001

## SUP-002 | Supplemental Assessments | Why did I get a supplemental bill after buying my home?
Your purchase price becomes your new base year value. If higher than the prior owner's assessed value, the county issues a supplemental assessment covering the portion of the fiscal year after your purchase. It is a one-time bill — not collected through most mortgage impound accounts.
Source: BOE Publication 26 · SUP-002

## SUP-003 | Supplemental Assessments | How is the supplemental assessment calculated?
The difference between new and prior assessed values is prorated by months remaining in the fiscal year (July 1 to June 30). An event in July results in a nearly full-year bill; an event in May results in a small one.
Source: BOE Publication 26 · SUP-003

## SUP-004 | Supplemental Assessments | Can I receive more than one supplemental bill?
Yes. If your event occurs between January 1 and May 31, you may receive two bills: one for the remainder of the current fiscal year and one for the following fiscal year.
Source: BOE Publication 26 · SUP-004

## SUP-006 | Supplemental Assessments | Is a supplemental bill the same as my regular tax bill?
No. Your regular annual bill covers the full fiscal year and is typically paid through your mortgage impound account. The supplemental bill is a separate, one-time bill that your lender usually does not pay. You are responsible for paying it directly.
Source: BOE Publication 26 · SUP-006

## SUP-007 | Supplemental Assessments | What if my supplemental assessment is too high?
You have the right to appeal. The deadline is 60 days from the date of the Notice of Supplemental Assessment — shorter than the regular roll appeal window. Contact your county Assessment Appeals Board to file.
Source: BOE Publication 26 · SUP-007

## EX-HO-001 | Homeowners Exemption | What is the Homeowners Exemption?
The Homeowners Exemption reduces your assessed value by $7,000, saving most homeowners roughly $70 per year. Available to California homeowners who own and occupy their home as their primary residence as of January 1. Renews automatically once filed.
Source: BOE Publication 2 · EX-HO-001

## EX-HO-002 | Homeowners Exemption | How do I apply for the Homeowners Exemption?
File form BOE-266 with your county Assessor's office. Filing by February 15 gives the full exemption. Filing between February 16 and December 10 gives 80% of the exemption for that year. The exemption renews automatically once granted.
Source: BOE Publication 2 · EX-HO-002

## EX-HO-003 | Homeowners Exemption | Who qualifies?
You must own the property and occupy it as your primary residence as of January 1. Applies to one property only. Renters, landlords, and owners of vacation or investment properties do not qualify.
Source: BOE Publication 2 · EX-HO-003

## EX-HO-004 | Homeowners Exemption | I just bought a home. Do I need to apply?
Yes. The exemption does not transfer automatically. File a new claim (BOE-266). The Assessor may mail you a form after your purchase records, but it is your responsibility to file.
Source: BOE Publication 2 · EX-HO-004

## EX-HO-005 | Homeowners Exemption | What if I move out or no longer qualify?
You must notify the Assessor by filing an Advice of Termination. If you fail to do so and continue receiving the exemption, the county will remove it and may add a 25% penalty on the escaped tax value. Notify the Assessor promptly when your primary residence changes.
Source: BOE Rule 135 · EX-HO-005

## EX-VET-DV-001 | Disabled Veterans Exemption | What is the Disabled Veterans Exemption?
A significant property tax benefit for veterans with a service-connected disability or their surviving spouses. Two tiers: base exemption (approx. $161,083 assessed value reduction) and low-income exemption (approx. $241,627). Both amounts are adjusted annually by the BOE.
Source: BOE Publication 149 · EX-VET-DV-001

## EX-VET-DV-002 | Disabled Veterans Exemption | Who qualifies?
Veterans honorably discharged with a 100% service-connected VA disability rating, or rated totally disabled. Also veterans who are blind in both eyes or have lost use of two or more limbs. Property must be the veteran's primary residence.
Source: BOE Publication 149 · EX-VET-DV-002

## EX-VET-DV-004 | Disabled Veterans Exemption | How do I apply?
File form BOE-261-G with your county Assessor's office, along with your DD-214 and VA disability rating letter. For the low-income tier, include income documentation. Must be renewed annually by February 15.
Source: BOE Publication 149 · EX-VET-DV-004

## EX-VET-DV-005 | Disabled Veterans Exemption | Does it need to be renewed every year?
Yes. Unlike the Homeowners Exemption, the Disabled Veterans Exemption requires annual renewal by February 15. The Assessor mails renewal forms in late fall, but filing is your responsibility.
Source: BOE Publication 149 · EX-VET-DV-005

## EX-VET-B-001 | Basic Veterans Exemption | What is the Basic Veterans Exemption?
Reduces assessed value by $4,000 (approx. $40/year savings). Available to veterans honorably discharged who served during a qualifying war period, and their unmarried surviving spouses. No disability rating required.
Source: BOE Publication 1 · EX-VET-B-001

## EX-VET-B-003 | Basic Veterans Exemption | How do I apply?
File form BOE-261 with your county Assessor's office along with your DD-214. File by February 15 for the full exemption. Renews automatically once granted.
Source: BOE Publication 1 · EX-VET-B-003

## BILL-001 | Billing and Payments | When are property tax bills mailed?
California county property tax bills are typically mailed in late October, covering the full fiscal year July 1 through June 30.
Source: BOE Publication 29 · BILL-001

## BILL-002 | Billing and Payments | When are property taxes due?
Two installments: First installment due November 1, delinquent after December 10. Second installment due February 1, delinquent after April 10. Both can be paid upfront if preferred.
Source: BOE Publication 29 · BILL-002

## BILL-003 | Billing and Payments | What happens if I miss a deadline?
A 10% penalty is added. If the second installment is not paid by June 30, the property becomes tax-defaulted with an additional $33 redemption fee. After five years of non-payment, the property may be subject to the county tax sale process.
Source: BOE Publication 29 · BILL-003

## BILL-004 | Billing and Payments | How can I pay my property taxes?
Online, by phone, by mail, or in person at your county Tax Collector's office. Check your county Tax Collector's website for specific payment options, addresses, and contact information.
Source: BOE Publication 29 · BILL-004

## BILL-009 | Billing and Payments | What is Mello-Roos?
Mello-Roos is a special tax for properties within a Community Facilities District (CFD), funding infrastructure like schools and roads. It appears as a separate line item on your bill and is not based on assessed value.
Source: BOE Publication 29 · BILL-009

## BILL-010 | Billing and Payments | Can I get a penalty waived?
Only in very limited circumstances: documented medical emergency, natural disaster, or county error. Financial hardship alone is not sufficient. You must pay the full amount including penalty first, then submit a penalty cancellation request with documentation to the Treasurer-Tax Collector.
Source: BOE Publication 29 · BILL-010

## APP-001 | Assessment Appeals | What is an assessment appeal?
A formal process to challenge the assessed value assigned to your property. If you believe your assessed value exceeds market value, you can appeal to the Assessment Appeals Board — an independent body separate from the Assessor. A successful appeal can reduce your taxes.
Source: BOE Publication 30 · APP-001

## APP-002 | Assessment Appeals | What are the grounds for appeal?
Most common: assessed value exceeds fair market value as of the applicable date. You may also appeal improperly denied exemptions, incorrect new construction valuations, or supplemental assessment values. You cannot appeal the tax rate or Mello-Roos charges.
Source: BOE Publication 30 · APP-002

## APP-003 | Assessment Appeals | What are the filing deadlines?
Regular roll: July 2 through November 30 of the assessment year. Supplemental assessments: 60 days from the Notice of Supplemental Assessment. Escape assessments: 60 days from notice. Missing a deadline generally forfeits your right to appeal for that year.
Source: BOE Publication 30 · APP-003

## APP-004 | Assessment Appeals | How do I file an appeal?
Submit an Application for Changed Assessment to your county Clerk of the Board of Supervisors. There is typically a filing fee. You must continue paying taxes on time during the appeal to avoid penalties. Contact your county Assessment Appeals Board for local filing instructions.
Source: BOE Publication 30 · APP-004

## APP-007 | Assessment Appeals | Do I keep paying taxes while my appeal is pending?
Yes. Filing an appeal does not suspend your payment obligation. If your appeal succeeds and the assessed value is reduced, you will receive a refund for any overpayment, with interest in some cases.
Source: BOE Publication 30 · APP-007

## P8-001 | Prop 8 Reductions | What is a Proposition 8 reduction?
If your property's current market value falls below its Prop 13 factored base year value, the county must temporarily reduce your assessed value to the lower market value. This is temporary — not a permanent change to your base year value.
Source: BOE Publication 9 · P8-001

## P8-002 | Prop 8 Reductions | How do I request a Prop 8 decline in value review?
If you believe your property's market value as of January 1 is lower than your current assessed value, contact your county Assessor's office to request an informal review. Most counties have a Decline in Value review form available online or at the office. You must request a review for the current assessment year only — you cannot apply retroactively for prior years.
Source: BOE Publication 800-10 · P8-002

## P8-003 | Prop 8 Reductions | What evidence supports a Prop 8 review?
Submit comparable sales — properties similar to yours that sold near January 1 of the tax year, in the same general location, with similar size and condition. A recent appraisal also works. The Assessor monitors market conditions and may reduce your value proactively, but you can initiate the process anytime. Only the most recent January 1 assessment may be reviewed.
Source: BOE Publication 800-10 · P8-003

## P8-004 | Prop 8 Reductions | Will my taxes go back up when the market recovers?
Yes. As the market recovers, your assessed value is restored up to the Prop 13 factored base year value — never above it. During recovery, your value can increase by more than 2% per year until it reaches the factored base year value.
Source: BOE Publication 9 · P8-004

## P8-005 | Prop 8 Reductions | Can assessed value increase by more than 2% after a Prop 8 reduction?
Yes. The standard 2% annual cap applies only when enrolled at the factored base year value. During recovery from a Prop 8 reduction, increases can exceed 2% per year until returning to the Prop 13 cap. Example: Property purchased at $450,000. Year 3, market drops to $350,000 — enrolled at $350,000 (Prop 8 reduction). Year 4, market recovers to $400,000 — enrolled at $400,000, a 14.3% increase. This is legal because the $400,000 is still below the Prop 13 factored base year value of approximately $477,000. Once market value reaches the factored base year value, the 2% cap resumes.
Source: BOE Publication 800-10 · P8-005

## P8-006 | Prop 8 Reductions | Can I formally appeal a Prop 8 determination?
If the Assessor disagrees with your position on market value, file a formal Assessment Appeal with your county Assessment Appeals Board. For a regular roll value, the filing window is July 2 through November 30. You must continue paying taxes during the appeal. If successful, you receive a refund for the overpaid amount.
Source: BOE Publication 800-10 · P8-006

## P19-001 | Proposition 19 | What is Proposition 19?
Proposition 19 was approved by California voters on November 3, 2020. It made two major changes: (1) New rules for the parent-child and grandparent-grandchild exclusion from reassessment, effective February 16, 2021; and (2) expanded base year value transfer rights for homeowners age 55 or older, severely disabled persons, and victims of wildfire or natural disaster, effective April 1, 2021. Prop 19 significantly tightened the family transfer exclusion by requiring the property to be used as the recipient's primary residence.
Source: BOE Publication 801 · P19-001

## P19-002 | Proposition 19 | Parent-child exclusion — basic rules
Under Prop 19 (effective February 16, 2021), a parent can transfer their primary residence to a child without triggering full reassessment — but only if the child moves in and makes it their primary residence within one year. The property must have been the parent's primary residence. Rentals, vacation homes, and investment properties no longer qualify. Transfers in either direction (parent to child or child to parent) are eligible.
Source: BOE Publication 800-1 · P19-002

## P19-003 | Proposition 19 | Parent-child exclusion — the $1 million value cap
The child keeps the parent's factored base year value only if the property's current market value does not exceed the parent's factored base year value plus $1,000,000 (adjusted periodically by the California House Price Index). For transfers from February 16, 2025 through February 15, 2027, the adjusted cap is $1,044,586. If market value exceeds the cap, the difference is added to the transferred taxable value.
Source: BOE Publication 800-1 · P19-003

## P19-004 | Proposition 19 | Parent-child exclusion — value cap calculation example
Example: Parent's factored base year value is $300,000. Market value at transfer is $1,500,000. Cap = $300,000 + $1,000,000 = $1,300,000. Market value exceeds cap by $200,000. Child's new taxable value = $300,000 + $200,000 = $500,000. Had market value been $1,100,000 (under cap), child would keep the $300,000 base unchanged and save approximately $8,000 per year.
Source: BOE Publication 801 · P19-004

## P19-005 | Proposition 19 | Parent-child exclusion — how to apply and deadlines
File form BOE-19-P with the County Assessor where the property is located. File within three years of the transfer date, or within six months of a supplemental or escape assessment notice. The child must also file for the Homeowners Exemption (BOE-266) or Disabled Veterans Exemption (BOE-261-G) within one year of the transfer date to receive the exclusion retroactive to the date of transfer. If the exemption is filed after one year, the exclusion only applies going forward. Contact your county Assessor's office for assistance.
Source: BOE Publication 800-1 · P19-005

## P19-006 | Proposition 19 | Grandparent-grandchild exclusion
Same rules as parent-child, but there is one additional requirement: the grandchild's parents who are children of the grandparent must be deceased before the transfer. The grandchild must move in as primary residence within one year. The same $1 million value cap applies. File form BOE-19-G with the County Assessor.
Source: BOE Publication 800-2 · P19-006

## P19-007 | Proposition 19 | Base year value transfer for seniors age 55 and older
Homeowners age 55 or older can sell their principal residence and transfer its factored base year value to a replacement home anywhere in California. Can be used up to three times. The replacement must be purchased or newly constructed within two years of selling the original. Must be age 55 at time of selling the original. If married, only one spouse needs to be 55. File form BOE-19-B with the County Assessor where the replacement property is located. Effective April 1, 2021.
Source: BOE Publication 800-3 · P19-007

## P19-008 | Proposition 19 | Senior base year value transfer — buying a more expensive home
You can buy a replacement of any value. If replacement costs more than original, the excess is added to your transferred base year value. Equal or lesser value thresholds: buying before you sell — replacement must be 100% or less of original sale price; buying within year 1 after selling — up to 105%; buying in year 2 — up to 110%. Beyond those thresholds the overage is added to your base year value.
Source: BOE Publication 800-3 · P19-008

## P19-009 | Proposition 19 | Base year value transfer for severely disabled persons
Severely and permanently disabled persons of any age can transfer their base year value to a replacement home anywhere in California, up to three times. Requirements are the same as the senior transfer except disability replaces the age requirement. File forms BOE-19-D and BOE-19-DC (Certificate of Disability) with the County Assessor where the replacement is located.
Source: BOE Publication 800-4 · P19-009

## P19-010 | Proposition 19 | Base year value transfer for wildfire and natural disaster victims
Property owners whose primary residence was substantially damaged (more than 50% of improvement value) by a Governor-declared wildfire or natural disaster can transfer their base year value to a replacement home anywhere in California, with no age requirement and no limit on number of uses. The replacement must be purchased or built within two years of the sale. File form BOE-19-V with the County Assessor where the replacement is located.
Source: BOE Publication 801 · P19-010

## P19-011 | Proposition 19 | What changed from the old parent-child exclusion?
Before Prop 19, parents could transfer both their primary residence and up to $1 million of other real property (rentals, vacation homes, investment property) without reassessment, with no value cap and no move-in requirement. Prop 19 eliminated that broader exclusion for all transfers on or after February 16, 2021. Now only primary residences qualify, and the child must actually live there. Investment and rental properties transferred to children after February 15, 2021 are fully reassessed.
Source: BOE Publication 800-1 · P19-011

## P19-012 | Proposition 19 | I inherited my parent's house — what do I need to do and when?
Time is critical. The date of death is the date of Change in Ownership — even if the property is still in trust or probate. You have one year from the date of death to move in and file for the Homeowners or Disabled Veterans Exemption to get the exclusion retroactive to the date of death. File form BOE-19-P within three years of the date of death. Also file form BOE-502-D (Change in Ownership Statement — Death of Real Property Owner) within 150 days of death. Contact your county Assessor's office as soon as possible.
Source: BOE Publication 800-9 · P19-012

## P19-013 | Proposition 19 | Can I buy my replacement home before I sell my original?
Yes. You can purchase or build the replacement first and sell the original afterward — as long as both transactions occur within two years of each other. The base year value transfer applies as of the date the original is sold. If you buy before selling, the equal-or-lesser-value threshold is 100% of the original's eventual sale price. Note: the transfer is not finalized until the original actually sells, so taxes on the replacement will initially be based on its purchase price and will be adjusted — with a refund if applicable — once the original sells and the claim is processed.
Source: BOE Rule 462.540 · P19-013

## P19-014 | Proposition 19 | Does the parent-child exclusion apply to family farms?
Yes. Under Prop 19 / Rule 462.520, the intergenerational transfer exclusion applies not only to a principal residence but also to a family farm — defined as land under cultivation, used for pasture or grazing, or used to produce agricultural commodities. The farm does not need to include a residence to qualify. The eligible transferee (child or grandchild) does not need to live on the farm, but must continue to use it as a family farm. If the property stops being used as a family farm by an eligible transferee, the exclusion is removed and the assessed value reverts to market value. File form BOE-19-P (parent-child) or BOE-19-G (grandparent-grandchild) with the County Assessor.
Source: BOE Rule 462.520 · P19-014

## P19-015 | Proposition 19 | What happens if my child moves out after inheriting my home under Prop 19?
The exclusion is lost. If the child stops using the inherited property as their primary residence, the intergenerational transfer exclusion is removed. The assessed value will then be based on the full market value that was established at the time of the original transfer, adjusted for inflation — not the low base year value the parent had. There is a one-year grace period: if another eligible transferee (such as a sibling who also qualifies) moves in within one year, the exclusion can be preserved. Callers should be aware that renting out the property, even temporarily, triggers removal of the exclusion.
Source: BOE Rule 462.520 · P19-015

## DEATH-001 | Death of Property Owner | What happens to property taxes when a property owner dies?
The death of a property owner is a Change in Ownership under California law, occurring as of the date of death — even if the property remains in trust or probate. This may trigger reassessment to market value unless an exclusion applies: transfers to a surviving spouse or registered domestic partner are automatic (no form required); Prop 19 parent-child and grandparent-grandchild exclusions require a form and have deadlines; the cotenant exclusion applies to qualifying co-owners.
Source: BOE Publication 800-9 · DEATH-001

## DEATH-002 | Death of Property Owner | What forms do I need to file when a property owner dies?
File form BOE-502-D (Change in Ownership Statement — Death of Real Property Owner) with the County Assessor within 150 days of the date of death. If the estate is in probate, file before or at the time the inventory and appraisal are filed with the court. This form notifies the Assessor and alerts you to any exclusion claim forms that also need to be filed, such as BOE-19-P (parent-child) or BOE-58-H (cotenant).
Source: BOE Publication 800-9 · DEATH-002

## DEATH-003 | Death of Property Owner | My parent left me their house in a trust. Do I still need to act quickly?
Yes — do not wait. Even if the property is in a trust, the date of death is the date of Change in Ownership for property tax purposes, not the date of distribution or deed transfer. The one-year clock to move in and file for the Homeowners Exemption starts at the date of death. If you miss the one-year window, you lose the ability to get the Prop 19 exclusion retroactive to the date of death.
Source: BOE Publication 800-9 · DEATH-003

## DEATH-004 | Death of Property Owner | Cotenant exclusion for co-owners who are not spouses
If two people (other than spouses or registered domestic partners) owned and lived together in a home as their primary residence for at least one year, and one dies, the surviving co-owner may qualify for the Cotenant Exclusion from reassessment. The surviving cotenant must inherit 100% of the property and sign an affidavit of continuous co-occupancy. File form BOE-58-H with the County Assessor. Applies to siblings, non-registered domestic partners, friends, or other co-owners.
Source: BOE Publication 800-8 · DEATH-004

## SR-001 | Senior Assistance | Property Tax Postponement Program for seniors age 62+
The California State Controller offers a Property Tax Postponement Program for homeowners age 62 or older. Qualifying homeowners can defer current-year property taxes on their primary residence — the State Controller pays the tax directly to the county on your behalf. Requirements: at least 40% equity in the home and annual household income below the program limit (approximately $49,000; check sco.ca.gov for the current year limit). Deferred taxes accrue interest and must be repaid when the home is sold, the owner moves out, or the owner dies without a qualifying successor.
Source: BOE Publication 800-5 · SR-001

## SR-002 | Senior Assistance | Payment plan for delinquent property taxes
If property taxes have been delinquent for less than five years, the County Tax Collector typically offers a five-year Permanent Installment Plan. At enrollment, pay 20% of the outstanding amount plus a processing fee; the remaining 80% is paid in equal installments over four years. Interest accrues at 1.5% per month (18% per year) on the unpaid balance. You must stay current on future annual tax bills to remain in good standing. Contact your county Treasurer-Tax Collector for details.
Source: BOE Publication 800-5 · SR-002
"""


def parse_knowledge_base(raw: str) -> list[dict]:
    chunks = []
    current = None
    for line in raw.strip().splitlines():
        line = line.strip()
        if line.startswith("## "):
            if current:
                chunks.append(current)
            parts = line[3:].split(" | ", 2)
            current = {
                "id": parts[0] if len(parts) > 0 else "",
                "topic": parts[1] if len(parts) > 1 else "",
                "question": parts[2] if len(parts) > 2 else "",
                "content": "",
                "source": ""
            }
        elif line.startswith("Source:") and current:
            current["source"] = line[7:].strip()
        elif current is not None:
            current["content"] = (current["content"] + " " + line).strip()
    if current:
        chunks.append(current)
    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r'[a-z]+', text.lower())

def build_tfidf_index(chunks: list[dict]) -> dict:
    corpus = []
    for c in chunks:
        text = f"{c['topic']} {c['question']} {c['content']}"
        corpus.append(tokenize(text))
    df = Counter()
    for doc in corpus:
        for term in set(doc):
            df[term] += 1
    N = len(corpus)
    idf = {term: math.log(N / freq) for term, freq in df.items()}
    vectors = []
    for doc in corpus:
        tf = Counter(doc)
        total = len(doc)
        vec = {term: (count / total) * idf.get(term, 0) for term, count in tf.items()}
        vectors.append(vec)
    return {"vectors": vectors, "idf": idf}

def cosine_similarity(a: dict, b: dict) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    mag_a = math.sqrt(sum(v**2 for v in a.values()))
    mag_b = math.sqrt(sum(v**2 for v in b.values()))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0

def retrieve(query: str, chunks: list[dict], index: dict, top_k: int = 4) -> list[dict]:
    idf = index["idf"]
    q_tokens = tokenize(query)
    tf = Counter(q_tokens)
    total = len(q_tokens)
    q_vec = {term: (count / total) * idf.get(term, 0) for term, count in tf.items()}
    scores = [cosine_similarity(q_vec, v) for v in index["vectors"]]
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in ranked[:top_k] if scores[i] > 0]


SYSTEM_BASE = """You are a California County Property Tax Assistant. You help property owners understand California property taxes using only the provided reference material.

Rules:
- Answer only from the REFERENCE MATERIAL below. Do not use outside knowledge.
- Be clear and friendly. Define technical terms when you use them.
- End every answer with the Source line from the matching reference entry.
- If the question is outside the reference material, say so and direct the user to contact their county Assessor's office.
- Never guess. If unsure, escalate to the office.
- Do not answer questions unrelated to California property taxes."""

def build_system_prompt(retrieved_chunks: list[dict]) -> str:
    if not retrieved_chunks:
        return SYSTEM_BASE + "\n\nNo relevant reference material found for this query."
    refs = "\n\n---\n\n".join(
        f"[{c['id']}] {c['topic']} — {c['question']}\n{c['content']}\nSource: {c['source']}"
        for c in retrieved_chunks
    )
    return f"{SYSTEM_BASE}\n\n=== REFERENCE MATERIAL ===\n\n{refs}"

def estimate_tokens(text: str) -> int:
    return len(text) // 4

def chat():
    client = Anthropic()
    chunks = parse_knowledge_base(RAW_KNOWLEDGE_BASE)
    index = build_tfidf_index(chunks)
    history = []

    print(f"\nLoaded {len(chunks)} knowledge base chunks.")
    print("California County Property Tax Assistant (RAG mode)")
    print("Model: claude-haiku-4-5  |  Type 'quit' to exit\n")
    print("-" * 60)

    session_input_tokens = 0
    session_output_tokens = 0

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        retrieved = retrieve(user_input, chunks, index, top_k=4)
        system_prompt = build_system_prompt(retrieved)
        chunk_ids = [c["id"] for c in retrieved]
        est_system_tokens = estimate_tokens(system_prompt)
        print(f"\n[RAG] Retrieved: {chunk_ids} | ~{est_system_tokens} system tokens")

        history.append({"role": "user", "content": user_input})

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            system=system_prompt,
            messages=history
        )

        reply = response.content[0].text
        history.append({"role": "assistant", "content": reply})

        session_input_tokens += response.usage.input_tokens
        session_output_tokens += response.usage.output_tokens

        cost = (session_input_tokens * 0.80 + session_output_tokens * 4.00) / 1_000_000
        print(f"\nAssistant: {reply}")
        print(f"\n[Tokens this session: {session_input_tokens} in / {session_output_tokens} out | Est. cost: ${cost:.5f}]")

    print("\nSession ended.")


if __name__ == "__main__":
    chat()
