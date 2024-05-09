import streamlit as st


st.set_page_config(page_title="Churninator | Öneriler", page_icon=":robot:", layout="centered")


st.markdown("# Öneriler")
st.sidebar.header("Öneriler")


with st.expander("***Credit Builder program***"):
    st.markdown("`CreditBuilder`")
    st.markdown("Aimed at customers with low credit scores, offering secured credit cards or credit-building loans with flexible terms and lower interest rates.")
    st.markdown("**Promotion:** Provide an exclusive discount on the annual fee or interest rate for customers who enroll in the 'Credit Rebuilder' program within a specified timeframe.")

with st.expander("***Financial Health Checkup***"):
    st.markdown("`Hibernating` `AbouttoSleep` `CreditBuilder`")
    st.markdown("Provide personalized financial consultations with a dedicated advisor.")
    st.markdown("**Promotion:** Offer a complimentary session to review their current financial situation, identify areas for improvement, and create a customized plan to achieve their financial goals.")
    st.markdown("**Objective:** Help customers in this segment regain control of their finances and prevent them from becoming dormant or churning.")


# * Total_Relationship_Count: (bunu herkese öner)
#     * Offer personalized incentives such as discounted interest rates or cashback rewards for opening additional accounts or signing up for new services.
#         * Offer discounted interest rates for opening additional accounts or signing up for new services.
#             * Hoşgeldin faizi
#         * Offer cashback rewards for opening additional accounts or signing up for new services.
#                 df.groupby("Total_Relationship_Count")["Target"].mean()
#                 # 1   0.256
#                 # 2   0.278
#                 # 3   0.174
#                 # 4   0.118
#                 # 5   0.120
#                 # 6   0.105
#
# * Borçlanmayı daha ucuza getirmeye yönelik kampanyalar:
#                 df.groupby("Target")["Avg_Utilization_Ratio"].mean() # TODO borcu düşük olanların churn etme oranı daha yüksek (x3.5).
#                 # 0   0.296
#                 # 1   0.162
#
#                 df.groupby("Target")["Total_Revolving_Bal"].mean() # TODO
#                 # 0   1256.604
#                 # 1    672.823
#
#
# * "Welcome Back" campaign
#     * Product/Service: Offer a limited-time promotion for reactivating their account, such as a cashback bonus or a waived annual fee on their credit card.
#     * Promotion: Send targeted emails or direct mailers inviting them to take advantage of exclusive offers upon reactivation.
#     * Objective: Encourage dormant customers to resume their banking activities by providing attractive incentives and reminders of the value of their relationship with the bank.
#         * #Hibernating #Low_Relationship_Count #AbouttoSleep
#
# * Credit Builder program
#     * Aimed at customers with low credit scores, offering secured credit cards or credit-building loans with flexible terms and lower interest rates.
#     * Promotion: Provide an exclusive discount on the annual fee or interest rate for customers who enroll in the "Credit Rebuilder" program within a specified timeframe.
#         * #CreditBuilder
#
# * “Family Bundle” package
#     * .
#         * #Family
#
# * “Career Starter” package
#     * Offering a checking account with budgeting tools, a student loan refinancing option, and access to career development resources.
#     * Promotion: Host networking events or workshops exclusively for young professionals who sign up for the "Career Starter" package, providing opportunities to connect with industry mentors or attend skill-building sessions.
#         * #YoungProf
#
# * Elite Rewards program
#     * Action: Maintain engagement and satisfaction by continuing to deliver exceptional service and rewards.
#         * Example: Send personalized thank-you notes or surprise gifts to top-tier customers to strengthen their loyalty further.
#     * Offer exclusive rewards and benefits, such as VIP customer service, premium travel perks, and enhanced cashback rates, for top-tier customers.
#     * Promotion: Invite customers in the Champions segment to enroll in the elite rewards program, showcasing the luxury and prestige associated with membership.
#     * Objective: Recognize and reward the loyalty of high-value customers, reinforcing their status as brand ambassadors and reducing the likelihood of churn.
#         * #Champions #CantLose
#
#
# * Q4’te Q1’e kıyasla daha fazla harcama yapmaları
# * Seasonal Spending Insights
# 	Total Transaction Amount Change Q4 over Q1:
#     * Offer: Send personalized insights or alerts to customers regarding their spending patterns and changes from quarter to quarter.
#     * Objective: Help customers understand their financial behavior and identify opportunities for optimizing their spending habits, thus enhancing loyalty.




st.markdown("<center>© 2024  DEG Bilgi Teknolojileri Danışmanlık ve Dağıtım A.Ş. Tüm hakları saklıdır.</center>", unsafe_allow_html=True)