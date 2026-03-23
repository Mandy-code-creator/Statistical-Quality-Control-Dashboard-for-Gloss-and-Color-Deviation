st.markdown("---")
    st.subheader("🎯 Smart Focus: High-Risk Gloss Codes (≥ 3 NG Batches)")
    st.caption("Strictly isolates paint codes where at least 3 distinct batches have exceeded Gloss limits (Lab or Line).")

    # LỌC DỮ LIỆU RÁC CHO BẢNG SMART FOCUS
    dff_focus = dff.dropna(subset=['Gloss_LSL', 'Gloss_USL', 'Gloss_Lab', 'Online_Gloss_Top']).copy()
    dff_focus = dff_focus[(dff_focus['Gloss_LSL'] > 0) & (dff_focus['Gloss_USL'] > 0) & (dff_focus['Gloss_Lab'] > 0) & (dff_focus['Online_Gloss_Top'] > 0)]

    if not dff_focus.empty:
        # BƯỚC 1: Đánh dấu NG CHỈ DÀNH CHO LỖI ĐỘ BÓNG
        dff_focus['Gloss_NG'] = (dff_focus['Gloss_Lab'] < dff_focus['Gloss_LSL']) | \
                                (dff_focus['Gloss_Lab'] > dff_focus['Gloss_USL']) | \
                                (dff_focus['Online_Gloss_Top'] < dff_focus['Gloss_LSL']) | \
                                (dff_focus['Online_Gloss_Top'] > dff_focus['Gloss_USL'])
        
        # BƯỚC 2: Tính tổng số Lô và tổng số Cuộn
        focus_df = dff_focus.groupby(['Ma_Son', 'Supplier']).agg(
            So_Lo=('Batch_Lot', 'nunique'),
            So_Cuon=('Batch_Lot', 'count')
        ).reset_index()

        # BƯỚC 3: Lọc lấy thông tin các mẻ bị NG độ bóng
        ng_data = dff_focus[dff_focus['Gloss_NG']]
        
        if not ng_data.empty:
            # Đếm số lô NG và Nối tên các lô NG lại
            ng_batches_info = ng_data.groupby(['Ma_Son', 'Supplier']).agg(
                NG_Batch_Count=('Batch_Lot', 'nunique'),
                Out_Of_Spec_Batches=('Batch_Lot', lambda x: ', '.join(x.dropna().astype(str).unique()))
            ).reset_index()

            # ---> CHỐT CHẶN MỚI: CHỈ LẤY CÁC MÃ CÓ TỪ 3 BATCH NG TRỞ LÊN <---
            ng_batches_info = ng_batches_info[ng_batches_info['NG_Batch_Count'] >= 3]

            if not ng_batches_info.empty:
                # BƯỚC 4: Kết hợp dữ liệu
                focus_df = pd.merge(focus_df, ng_batches_info, on=['Ma_Son', 'Supplier'], how='inner')

                # Tạo cột hiển thị tỷ lệ
                focus_df['Batch_Ratio'] = focus_df['NG_Batch_Count'].astype(str) + " / " + focus_df['So_Lo'].astype(str)

                # Sắp xếp: Mã nào có nhiều Lô hỏng nhất đưa lên đầu
                focus_df = focus_df.sort_values(by=['NG_Batch_Count', 'So_Cuon'], ascending=[False, False])

                # Dàn cột hiển thị
                focus_df_display = focus_df[['Ma_Son', 'Supplier', 'Batch_Ratio', 'So_Cuon', 'Out_Of_Spec_Batches']]
                focus_df_display.columns = ['Paint Code', 'Supplier', 'NG / Total Batches', 'Total Coils', 'Out of Spec Batches (Gloss)']

                st.dataframe(
                    focus_df_display,
                    use_container_width=True, 
                    hide_index=True
                )
                st.info("💡 **Actionable Insight:** These codes are highly unstable (≥ 3 NG Batches). Copy a `Paint Code` and check the **Gloss Analysis (SPC)** tab immediately to trace the root cause!")
            else:
                st.success("🎉 Excellent! No paint codes have 3 or more batches exceeding Gloss control limits.")
        else:
            st.success("🎉 Excellent! No paint codes have 3 or more batches exceeding Gloss control limits.")
    else:
        st.success("No valid data available for Smart Focus analysis.")
